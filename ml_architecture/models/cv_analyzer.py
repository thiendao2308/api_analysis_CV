from typing import List, Optional
from pydantic import BaseModel

class SectionAnalysis(BaseModel):
    """
    Kết quả phân tích cho một mục cụ thể trong CV.
    """
    section_name: str
    content: Optional[str] = None
    score: Optional[float] = None
    suggestions: List[str] = []

class KeywordAnalysis(BaseModel):
    """
    Chứa kết quả phân tích toàn diện về từ khóa/kỹ năng.
    """
    required_keywords: List[str]
    matched_keywords: List[str]
    missing_keywords: List[str]
    suggestions: List[str]

class CVAnalysisResult(BaseModel):
    """
    Mô hình dữ liệu cho kết quả phân tích CV cuối cùng.
    """
    overall_score: float
    strengths: List[str] = []
    section_analysis: List[SectionAnalysis]
    keyword_analysis: KeywordAnalysis # Thay thế keyword_matches bằng một đối tượng duy nhất

    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 85,
                "strengths": ["Bố cục CV rõ ràng, đầy đủ các mục quan trọng."],
                "section_analysis": [
                    {
                        "section_name": "Kinh nghiệm",
                        "content": "Làm việc tại công ty X...",
                        "score": None,
                        "suggestions": []
                    }
                ],
                "keyword_analysis": {
                    "required_keywords": ["Python", "JavaScript", "SQL"],
                    "matched_keywords": ["Python", "JavaScript"],
                    "missing_keywords": ["SQL"],
                    "suggestions": ["Bạn có thể bổ sung kinh nghiệm về SQL để phù hợp hơn."]
                }
            }
        } 