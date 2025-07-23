import os
import sys
import logging
import traceback
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from ml_architecture.services.llm_api_extractor_cv import extract_cv_info_from_text
from ml_architecture.services.llm_api_extractor_jd import extract_skills_from_jd
from ml_architecture.services.cv_evaluation_service import CVEvaluationService

# Add the ml_architecture directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_architecture'))
from ml_architecture.models.shared_models import CVAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CV Analysis API",
    description="API for analyzing CVs and matching with job descriptions",
    version="1.0.0"
)

# Khởi tạo analyzer
analyzer = CVEvaluationService()

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

# Initialize the CV analyzer
# analyzer = None

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

@app.post("/analyze-cv", response_model=CVAnalysisResult)
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
    if not analyzer:
         raise HTTPException(
            status_code=500,
            detail="Lỗi server: CV Analyzer không khả dụng. Vui lòng kiểm tra logs."
        )
        
    try:
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

        # LLM extraction cho CV
        llm_cv_result = extract_cv_info_from_text(cv_content)
        logging.info("[LLM Extraction] CV: %s", llm_cv_result)

        # LLM extraction cho JD (KHÔNG GỌI TRONG analyze_cv)
        # llm_jd_result = extract_skills_from_jd(jd_text)
        # logging.info("[LLM Extraction] JD: %s", llm_jd_result)

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
    if not analyzer:
        raise HTTPException(
            status_code=500,
            detail="Lỗi server: CV Analyzer không khả dụng."
        )
    
    try:
        # Trích xuất skills từ JD
        jd_skills = extract_skills_from_jd(jd_text)
        
        # Phân tích chi tiết JD
        from ml_architecture.data.jd_analysis_system import JDAnalysisSystem
        jd_analyzer = JDAnalysisSystem()
        jd_analysis = jd_analyzer.analyze_single_jd(jd_text)
        
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
        skills = extract_skills_from_jd(jd_text)
        return {"skills": skills}
    except Exception as e:
        logger.error(f"LLM API extraction failed: {e}")
        raise HTTPException(status_code=500, detail="LLM API extraction failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "analyzer_available": analyzer is not None
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 