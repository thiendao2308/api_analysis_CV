from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import uvicorn
import logging
import traceback

from .services.file_processor import process_cv_file
from .models.cv_jd_matcher import DetailedCVAnalyzer
from .models.cv_analyzer import CVAnalysisResult

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI CV Analyzer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo analyzer một lần khi server khởi động để tối ưu hiệu năng
# Tránh việc phải tải lại model SpaCy cho mỗi request
try:
    analyzer = DetailedCVAnalyzer()
    logger.info("DetailedCVAnalyzer đã được khởi tạo thành công.")
except Exception as e:
    logger.error(f"LỖI NGHIÊM TRỌNG: Không thể khởi tạo DetailedCVAnalyzer. Server có thể không hoạt động đúng. Lỗi: {e}")
    analyzer = None

@app.post("/analyze-cv", response_model=CVAnalysisResult)
async def analyze_cv(
    cv_file: UploadFile = File(...),
    jd_text: str = Form(...),
    job_requirements: Optional[str] = Form(None)
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

        if not cv_content or not jd_text:
            raise HTTPException(
                status_code=400,
                detail="Nội dung CV hoặc Mô tả công việc không được để trống"
            )

        # Phân tích CV với JD bằng bộ phân tích chi tiết mới
        try:
            logger.info("Bắt đầu phân tích chi tiết CV và JD/JR...")
            analysis_result = analyzer.analyze(
                cv_content=cv_content,
                jd_text=jd_text,
                jr_text=job_requirements
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
            detail=f"Lỗi không xác định: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Kiểm tra trạng thái hoạt động của server
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 