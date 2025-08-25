from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ParsedCV(BaseModel):
    """
    Cấu trúc dữ liệu chứa thông tin đã được bóc tách từ một CV.
    """
    job_title: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = []
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    projects: List[Dict[str, Any]] = []
    sections: Dict[str, str] = {}

class MLInsights(BaseModel):
    """
    Thông tin insights từ machine learning model
    """
    ml_score: Optional[float] = None
    important_features: Optional[List[str]] = None
    confidence: Optional[float] = None

class QualityAnalysis(BaseModel):
    """
    Phân tích chất lượng CV
    """
    quality_score: Optional[float] = None
    structure_score: Optional[float] = None
    content_score: Optional[float] = None
    presentation_score: Optional[float] = None
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None

class CVAnalysis(BaseModel):
    """
    Thông tin phân tích CV
    """
    job_title: Optional[str] = None
    skills: List[str] = []
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    projects: List[Dict[str, Any]] = []
    sections: Dict[str, str] = {}

class JDAnalysis(BaseModel):
    """
    Thông tin phân tích JD
    """
    extracted_skills: List[str] = []
    jd_text: Optional[str] = None

class MatchingAnalysis(BaseModel):
    """
    Phân tích matching giữa CV và JD
    """
    matching_skills: List[str] = []
    missing_skills: List[str] = []
    skills_match_score: float = 0.0
    position_match_score: int = 0
    total_skills_cv: int = 0
    total_skills_jd: int = 0

class Scores(BaseModel):
    """
    Điểm số đánh giá
    """
    ats_score: int = 0
    overall_score: int = 0

class CVAnalysisResult(BaseModel):
    """
    Kết quả phân tích CV hoàn chỉnh với cấu trúc mới
    """
    cv_analysis: Optional[CVAnalysis] = None
    jd_analysis: Optional[JDAnalysis] = None
    matching_analysis: Optional[MatchingAnalysis] = None
    quality_analysis: Optional[Dict[str, Any]] = None
    ml_insights: Optional[Dict[str, Any]] = None
    scores: Optional[Scores] = None
    feedback: Optional[str] = None
    suggestions: Optional[List[str]] = None
    job_category: Optional[str] = None
    job_position: Optional[str] = None
    error: Optional[str] = None
    analysis_timestamp: Optional[str] = None 