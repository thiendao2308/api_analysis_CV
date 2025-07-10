from typing import Dict, Any, List, Tuple
from ..models.shared_models import ParsedCV

class CVQualityAnalyzer:
    """
    Analyzes the structural quality of a CV.
    """
    def __init__(self):
        # Các mục tiêu chuẩn trong một CV
        self.required_sections = {
            "summary": "Tóm tắt/Mục tiêu nghề nghiệp",
            "skills": "Kỹ năng",
            "experience": "Kinh nghiệm làm việc",
            "education": "Học vấn"
        }

    def analyze(self, parsed_cv: ParsedCV) -> Dict[str, Any]:
        """
        Analyzes the CV's layout and completeness.

        Args:
            parsed_cv: The parsed CV data.

        Returns:
            A dictionary containing the quality score and a list of strengths.
        """
        score, strengths = self._check_layout_structure(parsed_cv)

        return {
            "quality_score": score,
            "strengths": strengths
        }

    def _check_layout_structure(self, parsed_cv: ParsedCV) -> Tuple[float, List[str]]:
        """
        Checks for the presence of standard CV sections.
        """
        found_sections_count = 0
        
        # Kiểm tra sự tồn tại của các mục
        if parsed_cv.summary:
            found_sections_count += 1
        if parsed_cv.skills:
            found_sections_count += 1
        if parsed_cv.experience:
            found_sections_count += 1
        if parsed_cv.education:
            found_sections_count += 1
            
        total_sections = len(self.required_sections)
        score = found_sections_count / total_sections
        
        strengths = []
        if score >= 0.75: # Nếu tìm thấy ít nhất 3/4 mục
            strengths.append("Bố cục CV rõ ràng, trình bày đầy đủ các mục quan trọng, tạo ấn tượng tốt về sự chuyên nghiệp.")
        elif score >= 0.5: # Nếu tìm thấy ít nhất 2/4 mục
            strengths.append("CV có cấu trúc tương đối, tuy nhiên có thể cải thiện bằng cách bổ sung thêm các mục còn thiếu để tăng tính thuyết phục.")

        return score, strengths 