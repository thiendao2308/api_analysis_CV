import os
import sys
import logging
import traceback
import gc
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import memory management
try:
    from ml_architecture.config.memory_config import MemoryManager, check_memory_usage
    # Apply memory optimizations for deployment
    MemoryManager.optimize_for_deployment()
except ImportError:
    # Fallback if memory config is not available
    class MemoryManager:
        @staticmethod
        def log_memory_usage(stage: str):
            pass
        @staticmethod
        def force_garbage_collection():
            gc.collect()
    
    def check_memory_usage():
        return True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CV Analysis API",
    description="API for analyzing CVs and matching with job descriptions",
    version="1.0.0"
)

# Lazy loading - chỉ khởi tạo analyzer khi cần
analyzer = None

from fastapi import Request
@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        import traceback
        print("=== EXCEPTION CAUGHT BY MIDDLEWARE ===")
        traceback.print_exc()
        raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_analyzer():
    """Lazy loading cho analyzer để tiết kiệm memory"""
    global analyzer
    if analyzer is None:
        try:
            # Log memory before loading analyzer
            MemoryManager.log_memory_usage("before_analyzer_init")
            
            # Force garbage collection before loading
            MemoryManager.force_garbage_collection()
            
            from ml_architecture.services.cv_evaluation_service import CVEvaluationService
            analyzer = CVEvaluationService()
            
            # Log memory after loading analyzer
            MemoryManager.log_memory_usage("after_analyzer_init")
            
            logger.info("✅ CV Analyzer đã được khởi tạo thành công")
        except Exception as e:
            logger.error(f"❌ Lỗi khi khởi tạo CV Analyzer: {e}")
            # Force garbage collection on error
            MemoryManager.force_garbage_collection()
            raise HTTPException(
                status_code=500,
                detail="Lỗi server: Không thể khởi tạo CV Analyzer"
            )
    
    # Double check analyzer is not None
    if analyzer is None:
        logger.error("❌ Analyzer is still None after initialization attempt")
        raise HTTPException(
            status_code=500,
            detail="Lỗi server: Analyzer không được khởi tạo"
        )
    
    return analyzer

async def process_cv_file(cv_file: UploadFile) -> str:
    """Process CV file and extract text content"""
    try:
        # Read file content
        content = await cv_file.read()
        
        # Check file type and process accordingly
        file_extension = cv_file.filename.lower().split('.')[-1] if cv_file.filename else ''
        
        if file_extension == 'pdf':
            # For PDF files, try to extract text with better error handling
            try:
                import PyPDF2
                import io
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                cv_text = ""
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cv_text += page_text + "\n"
                    except Exception as page_error:
                        logger.warning(f"Error extracting text from page: {page_error}")
                        continue
                return cv_text.strip() if cv_text.strip() else "PDF content could not be extracted"
            except Exception as pdf_error:
                logger.warning(f"PDF parsing failed: {pdf_error}, trying alternative method")
                try:
                    # Try alternative PDF parsing
                    import fitz  # PyMuPDF
                    doc = fitz.open(stream=content, filetype="pdf")
                    cv_text = ""
                    for page in doc:
                        cv_text += page.get_text() + "\n"
                    doc.close()
                    return cv_text.strip() if cv_text.strip() else "PDF content could not be extracted"
                except ImportError:
                    logger.warning("PyMuPDF not available")
                except Exception as fitz_error:
                    logger.warning(f"PyMuPDF parsing failed: {fitz_error}")
                
                # Fallback: return a placeholder text
                return "PDF content could not be extracted. Please try uploading a text file instead."
        elif file_extension in ['txt', 'docx']:
            # For text files, decode as UTF-8 with error handling
            return content.decode('utf-8', errors='ignore')
        else:
            # Default: try to decode as text
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Error processing CV file: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing CV file: {str(e)}")

@app.post("/analyze-cv")
async def analyze_cv(
    cv_file: UploadFile = File(...),
    job_category: str = Form(...),
    job_position: str = Form(...),
    jd_text: str = Form(...),
    job_requirements: str = Form(None)
):
    """
    Phân tích CV từ file và đánh giá mức độ phù hợp với JD/JR
    """
    try:
        # Check memory usage before processing
        if not check_memory_usage():
            raise HTTPException(
                status_code=503,
                detail="Server is under high memory load. Please try again later."
            )
        
        # Lazy load analyzer
        analyzer = get_analyzer()
        
        # Xử lý file CV để lấy nội dung text
        try:
            logger.info(f"Bắt đầu xử lý file: {cv_file.filename}")
            cv_content = await process_cv_file(cv_file)
            logger.info("Xử lý file CV thành công, thu được nội dung text.")
        except Exception as e:
            logger.error(f"Lỗi khi xử lý file CV: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=400,
                detail=f"Lỗi không thể xử lý file CV: {str(e)}"
            )

        if not cv_content:
            raise HTTPException(
                status_code=400,
                detail="Nội dung CV không được để trống"
            )

        # LLM extraction cho CV - lazy load
        try:
            from ml_architecture.services.llm_api_extractor_cv import extract_cv_info_from_text
            llm_cv_result = extract_cv_info_from_text(cv_content)
            logging.info("[LLM Extraction] CV: %s", llm_cv_result)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            llm_cv_result = {}

        # Phân tích CV với JD bằng bộ phân tích chi tiết mới
        try:
            logger.info("Bắt đầu phân tích chi tiết CV và JD...")
            logger.info(f"Job Category: {job_category}")
            logger.info(f"Job Position: {job_position}")
            
            analysis_result = analyzer.evaluate_cv_comprehensive(
                cv_text=cv_content,
                job_category=job_category,
                job_position=job_position,
                jd_text=jd_text,
                job_requirements=job_requirements
            )
            logger.info("Phân tích chi tiết hoàn tất.")
            
            # Clean up memory
            MemoryManager.force_garbage_collection()
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích CV: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi không xác định: {str(e)}"
            )

        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi không xác định: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi server: {str(e)}"
        )

@app.post("/analyze-jd")
async def analyze_jd(jd_text: str = Form(...)):
    """
    Phân tích JD và trích xuất skills
    """
    try:
        # Check memory usage before processing
        if not check_memory_usage():
            raise HTTPException(
                status_code=503,
                detail="Server is under high memory load. Please try again later."
            )
        
        # Lazy load analyzer
        analyzer = get_analyzer()
        
        # Trích xuất skills từ JD - lazy load
        try:
            from ml_architecture.services.llm_api_extractor_jd import extract_skills_from_jd
            jd_skills = extract_skills_from_jd(jd_text)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            jd_skills = []
        
        # Phân tích chi tiết JD - lazy load
        try:
            from ml_architecture.data.jd_analysis_system import JDAnalysisSystem
            jd_analyzer = JDAnalysisSystem()
            jd_analysis = jd_analyzer.analyze_single_jd(jd_text)
        except Exception as e:
            logger.warning(f"JD analysis failed: {e}")
            jd_analysis = {}
        
        # Clean up memory
        MemoryManager.force_garbage_collection()
        
        return {
            "jd_text": jd_text,
            "extracted_skills": jd_skills,
            "analysis": jd_analysis
        }
    except Exception as e:
        logger.error(f"Lỗi khi phân tích JD: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích JD: {str(e)}"
        )

@app.post("/extract-jd-skills-api")
async def extract_jd_skills_api(jd_text: str = Form(...)):
    """Extract skills from JD using LLM API (OpenAI)"""
    try:
        from ml_architecture.services.llm_api_extractor_jd import extract_skills_from_jd
        skills = extract_skills_from_jd(jd_text)
        return {"skills": skills}
    except Exception as e:
        logger.error(f"LLM API extraction failed: {e}")
        raise HTTPException(status_code=500, detail="LLM API extraction failed")

@app.get("/health")
async def health_check():
    """Health check endpoint - simplified for faster response"""
    try:
        memory_usage = MemoryManager.get_memory_usage()
        return {
            "status": "healthy",
            "memory_usage_mb": round(memory_usage, 2),
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CV Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze_cv": "/analyze-cv",
            "analyze_jd": "/analyze-jd",
            "health": "/health"
        }
    }

# Add new models for web client integration
class UserCVRequest(BaseModel):
    user_id: str
    cv_id: str
    job_category: str
    job_position: str = None

class JobPostingRequest(BaseModel):
    job_id: str
    job_title: str
    job_description: str
    job_requirements: str = None
    company_name: str = None

class WebClientAnalysisRequest(BaseModel):
    user_id: str
    cv_id: str
    job_id: str
    job_category: str
    job_position: str = None

@app.post("/analyze-cv-from-web-client")
async def analyze_cv_from_web_client(request: WebClientAnalysisRequest):
    """
    Phân tích CV từ web client - luồng mới
    CV và JD được import tự động từ web client
    """
    try:
        # Check memory usage before processing - but don't fail immediately
        memory_ok = check_memory_usage()
        if not memory_ok:
            logger.warning("High memory usage detected, but continuing...")
        
        # Lazy load analyzer - with better error handling
        try:
            analyzer = get_analyzer()
            if analyzer is None:
                raise HTTPException(
                    status_code=500,
                    detail="Lỗi server: Không thể khởi tạo analyzer"
                )
        except Exception as e:
            logger.error(f"❌ Lỗi khi khởi tạo analyzer: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi server: {str(e)}"
            )
        
        # 1. Lấy CV content từ web client database
        cv_content = await get_cv_content_from_web_client(request.user_id, request.cv_id)
        
        # 2. Lấy JD content từ web client database  
        jd_content = await get_jd_content_from_web_client(request.job_id)
        
        # 3. Phân tích CV với JD
        analysis_result = analyzer.evaluate_cv_comprehensive(
            cv_text=cv_content,
            job_category=request.job_category,
            job_position=request.job_position,
            jd_text=jd_content,
            job_requirements=None  # Có thể thêm từ JD content
        )
        
        # 4. Lưu kết quả phân tích vào web client database
        await save_analysis_result_to_web_client(
            user_id=request.user_id,
            cv_id=request.cv_id,
            job_id=request.job_id,
            analysis_result=analysis_result
        )
        
        # Clean up memory
        MemoryManager.force_garbage_collection()
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi phân tích CV từ web client: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi server: {str(e)}"
        )

@app.get("/get-user-cv/{user_id}/{cv_id}")
async def get_user_cv(user_id: str, cv_id: str):
    """
    Lấy thông tin CV của user từ web client
    """
    try:
        cv_info = await get_cv_info_from_web_client(user_id, cv_id)
        return cv_info
    except Exception as e:
        logger.error(f"Lỗi khi lấy CV: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy CV: {str(e)}"
        )

@app.get("/get-job-posting/{job_id}")
async def get_job_posting(job_id: str):
    """
    Lấy thông tin job posting từ web client
    """
    try:
        job_info = await get_job_info_from_web_client(job_id)
        return job_info
    except Exception as e:
        logger.error(f"Lỗi khi lấy job posting: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy job posting: {str(e)}"
        )

@app.post("/analyze-cv-with-job")
async def analyze_cv_with_job(request: UserCVRequest):
    """
    Phân tích CV với job category được chọn
    CV được lấy từ user account, JD được suggest tự động
    """
    try:
        # Check memory usage before processing - but don't fail immediately
        memory_ok = check_memory_usage()
        if not memory_ok:
            logger.warning("High memory usage detected, but continuing...")
        
        # Lazy load analyzer - with better error handling
        try:
            analyzer = get_analyzer()
            if analyzer is None:
                raise HTTPException(
                    status_code=500,
                    detail="Lỗi server: Không thể khởi tạo analyzer"
                )
        except Exception as e:
            logger.error(f"❌ Lỗi khi khởi tạo analyzer: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi server: {str(e)}"
            )
        
        # 1. Lấy CV content
        cv_content = await get_cv_content_from_web_client(request.user_id, request.cv_id)
        
        # 2. Tìm job postings phù hợp với category
        suggested_jobs = await get_suggested_jobs_for_category(request.job_category)
        
        # 3. Phân tích với từng job suggestion
        analysis_results = []
        for job in suggested_jobs[:3]:  # Top 3 suggestions
            jd_content = await get_jd_content_from_web_client(job['job_id'])
            
            analysis_result = analyzer.evaluate_cv_comprehensive(
                cv_text=cv_content,
                job_category=request.job_category,
                job_position=request.job_position or job.get('job_title'),
                jd_text=jd_content
            )
            
            analysis_results.append({
                "job_id": job['job_id'],
                "job_title": job['job_title'],
                "company_name": job.get('company_name'),
                "analysis": analysis_result
            })
        
        return {
            "user_id": request.user_id,
            "cv_id": request.cv_id,
            "job_category": request.job_category,
            "suggested_analyses": analysis_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi phân tích CV với job suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi server: {str(e)}"
        )

# Import web client integration
from ml_architecture.services.web_client_integration import WebClientIntegration, MockWebClientIntegration

# Initialize web client integration
web_client = MockWebClientIntegration()  # Use mock for development

# Helper functions for web client integration
async def get_cv_content_from_web_client(user_id: str, cv_id: str) -> str:
    """
    Lấy CV content từ web client database
    """
    logger.info(f"Getting CV content for user {user_id}, CV {cv_id}")
    return await web_client.get_cv_content(user_id, cv_id)

async def get_jd_content_from_web_client(job_id: str) -> str:
    """
    Lấy JD content từ web client database
    """
    logger.info(f"Getting JD content for job {job_id}")
    job_data = await web_client.get_job_content(job_id)
    return job_data.get('job_description', '') + '\n\n' + job_data.get('job_requirements', '')

async def get_cv_info_from_web_client(user_id: str, cv_id: str) -> dict:
    """
    Lấy thông tin CV từ web client
    """
    user_cvs = await web_client.get_user_cvs(user_id)
    for cv in user_cvs:
        if cv.get('cv_id') == cv_id:
            return cv
    return {}

async def get_job_info_from_web_client(job_id: str) -> dict:
    """
    Lấy thông tin job posting từ web client
    """
    return await web_client.get_job_content(job_id)

async def get_suggested_jobs_for_category(job_category: str) -> list:
    """
    Lấy danh sách job suggestions cho category
    """
    return await web_client.get_jobs_by_category(job_category)

async def save_analysis_result_to_web_client(user_id: str, cv_id: str, job_id: str, analysis_result: dict):
    """
    Lưu kết quả phân tích vào web client database
    """
    logger.info(f"Saving analysis result for user {user_id}, CV {cv_id}, job {job_id}")
    return await web_client.save_analysis_result(user_id, cv_id, job_id, analysis_result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 