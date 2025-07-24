from typing import Dict, Any, List, Tuple
from ml_architecture.models.shared_models import ParsedCV

class CVQualityAnalyzer:
    """
    BƯỚC 6: Analyzes the structural quality of a CV.
    Chấm điểm tổng thể ATS (MML)
    """
    def __init__(self):
        # Các mục tiêu chuẩn trong một CV
        self.required_sections = {
            "summary": "Tóm tắt/Mục tiêu nghề nghiệp",
            "skills": "Kỹ năng",
            "experience": "Kinh nghiệm làm việc",
            "education": "Học vấn"
        }
        
        # Tiêu chí đánh giá chất lượng
        self.quality_criteria = {
            "structure": {
                "weight": 0.3,
                "criteria": ["has_clear_structure", "has_professional_format", "has_consistent_sections"]
            },
            "content": {
                "weight": 0.4,
                "criteria": ["has_relevant_experience", "has_appropriate_skills", "has_education_info"]
            },
            "presentation": {
                "weight": 0.3,
                "criteria": ["has_professional_language", "has_no_grammar_errors", "has_good_length"]
            }
        }

    def analyze(self, parsed_cv: ParsedCV) -> Dict[str, Any]:
        """
        BƯỚC 6: Analyzes the CV's layout and completeness.

        Args:
            parsed_cv: The parsed CV data.

        Returns:
            A dictionary containing the quality score and analysis details.
        """
        print("🔍 BƯỚC 6: BẮT ĐẦU PHÂN TÍCH CHẤT LƯỢNG CV")
        
        # Phân tích cấu trúc
        structure_score, structure_details = self._analyze_structure(parsed_cv)
        
        # Phân tích nội dung
        content_score, content_details = self._analyze_content(parsed_cv)
        
        # Phân tích trình bày
        presentation_score, presentation_details = self._analyze_presentation(parsed_cv)
        
        # Tính điểm tổng hợp
        overall_score = (
            structure_score * self.quality_criteria["structure"]["weight"] +
            content_score * self.quality_criteria["content"]["weight"] +
            presentation_score * self.quality_criteria["presentation"]["weight"]
        )
        
        # Tạo strengths và weaknesses
        strengths = self._identify_strengths(structure_details, content_details, presentation_details)
        weaknesses = self._identify_weaknesses(structure_details, content_details, presentation_details)
        
        result = {
            "quality_score": overall_score,
            "structure_score": structure_score,
            "content_score": content_score,
            "presentation_score": presentation_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "details": {
                "structure": structure_details,
                "content": content_details,
                "presentation": presentation_details
            }
        }
        
        print(f"✅ BƯỚC 6: HOÀN THÀNH PHÂN TÍCH CHẤT LƯỢNG - Điểm: {overall_score:.2f}")
        return result

    def _analyze_structure(self, parsed_cv: ParsedCV) -> Tuple[float, Dict]:
        """BƯỚC 6: Phân tích cấu trúc CV"""
        details = {}
        score = 0.0
        
        # Kiểm tra các mục bắt buộc
        found_sections = 0
        total_sections = len(self.required_sections)
        
        if parsed_cv.summary:
            found_sections += 1
            details["has_summary"] = True
        else:
            details["has_summary"] = False
            
        if parsed_cv.skills:
            found_sections += 1
            details["has_skills"] = True
        else:
            details["has_skills"] = False
            
        if parsed_cv.experience:
            found_sections += 1
            details["has_experience"] = True
        else:
            details["has_experience"] = False
            
        if parsed_cv.education:
            found_sections += 1
            details["has_education"] = True
        else:
            details["has_education"] = False
        
        # Tính điểm cấu trúc
        structure_ratio = found_sections / total_sections
        details["structure_completeness"] = structure_ratio
        
        # Điểm cho cấu trúc rõ ràng
        if structure_ratio >= 0.75:
            details["has_clear_structure"] = True
            score += 0.4
        elif structure_ratio >= 0.5:
            details["has_clear_structure"] = True
            score += 0.2
        else:
            details["has_clear_structure"] = False
        
        # Điểm cho format chuyên nghiệp
        details["has_professional_format"] = True  # Giả định format tốt
        score += 0.3
        
        # Điểm cho sections nhất quán
        details["has_consistent_sections"] = True  # Giả định nhất quán
        score += 0.3
        
        return min(score, 1.0), details

    def _analyze_content(self, parsed_cv: ParsedCV) -> Tuple[float, Dict]:
        """BƯỚC 6: Phân tích nội dung CV"""
        details = {}
        score = 0.0
        
        # Điểm cho kinh nghiệm liên quan
        if parsed_cv.experience:
            details["has_relevant_experience"] = True
            score += 0.4
        else:
            details["has_relevant_experience"] = False
        
        # Điểm cho kỹ năng phù hợp
        if parsed_cv.skills and len(parsed_cv.skills) >= 3:
            details["has_appropriate_skills"] = True
            score += 0.4
        elif parsed_cv.skills:
            details["has_appropriate_skills"] = True
            score += 0.2
        else:
            details["has_appropriate_skills"] = False
        
        # Điểm cho thông tin học vấn
        if parsed_cv.education:
            details["has_education_info"] = True
            score += 0.2
        else:
            details["has_education_info"] = False
        
        return min(score, 1.0), details

    def _analyze_presentation(self, parsed_cv: ParsedCV) -> Tuple[float, Dict]:
        """BƯỚC 6: Phân tích trình bày CV"""
        details = {}
        score = 0.0
        
        # Giả định ngôn ngữ chuyên nghiệp (cần cải thiện với NLP)
        details["has_professional_language"] = True
        score += 0.4
        
        # Giả định không có lỗi ngữ pháp (cần cải thiện với grammar checker)
        details["has_no_grammar_errors"] = True
        score += 0.3
        
        # Điểm cho độ dài phù hợp
        total_length = len(str(parsed_cv))
        if 500 <= total_length <= 2000:  # Độ dài phù hợp
            details["has_good_length"] = True
            score += 0.3
        elif total_length < 500:
            details["has_good_length"] = False
            score += 0.1
        else:
            details["has_good_length"] = False
            score += 0.2
        
        return min(score, 1.0), details

    def _identify_strengths(self, structure_details: Dict, content_details: Dict, presentation_details: Dict) -> List[str]:
        """BƯỚC 6: Xác định điểm mạnh của CV"""
        strengths = []
        
        # Điểm mạnh về cấu trúc
        if structure_details.get("has_clear_structure", False):
            strengths.append("Cấu trúc CV rõ ràng và chuyên nghiệp")
        
        if structure_details.get("structure_completeness", 0) >= 0.75:
            strengths.append("Đầy đủ các mục quan trọng trong CV")
        
        # Điểm mạnh về nội dung
        if content_details.get("has_relevant_experience", False):
            strengths.append("Có kinh nghiệm làm việc phù hợp")
        
        if content_details.get("has_appropriate_skills", False):
            strengths.append("Có các kỹ năng chuyên môn phù hợp")
        
        if content_details.get("has_education_info", False):
            strengths.append("Thông tin học vấn đầy đủ")
        
        # Điểm mạnh về trình bày
        if presentation_details.get("has_professional_language", False):
            strengths.append("Sử dụng ngôn ngữ chuyên nghiệp")
        
        if presentation_details.get("has_good_length", False):
            strengths.append("Độ dài CV phù hợp")
        
        return strengths

    def _identify_weaknesses(self, structure_details: Dict, content_details: Dict, presentation_details: Dict) -> List[str]:
        """BƯỚC 6: Xác định điểm yếu của CV"""
        weaknesses = []
        
        # Điểm yếu về cấu trúc
        if not structure_details.get("has_clear_structure", False):
            weaknesses.append("Cấu trúc CV chưa rõ ràng")
        
        if structure_details.get("structure_completeness", 0) < 0.5:
            weaknesses.append("Thiếu nhiều mục quan trọng trong CV")
        
        # Điểm yếu về nội dung
        if not content_details.get("has_relevant_experience", False):
            weaknesses.append("Thiếu thông tin kinh nghiệm làm việc")
        
        if not content_details.get("has_appropriate_skills", False):
            weaknesses.append("Thiếu hoặc ít kỹ năng chuyên môn")
        
        if not content_details.get("has_education_info", False):
            weaknesses.append("Thiếu thông tin học vấn")
        
        # Điểm yếu về trình bày
        if not presentation_details.get("has_good_length", False):
            weaknesses.append("Độ dài CV không phù hợp")
        
        return weaknesses

    def calculate_ats_score(self, quality_analysis: Dict) -> int:
        """BƯỚC 6: Tính điểm ATS dựa trên phân tích chất lượng"""
        ats_score = 0
        
        # Điểm cho chất lượng tổng thể (40%)
        quality_score = quality_analysis.get("quality_score", 0)
        ats_score += int(quality_score * 40)
        
        # Điểm cho cấu trúc (30%)
        structure_score = quality_analysis.get("structure_score", 0)
        ats_score += int(structure_score * 30)
        
        # Điểm cho nội dung (30%)
        content_score = quality_analysis.get("content_score", 0)
        ats_score += int(content_score * 30)
        
        return min(ats_score, 100)

# Test function
if __name__ == "__main__":
    from ..models.shared_models import ParsedCV
    
    # Test với CV mẫu
    analyzer = CVQualityAnalyzer()
    
    sample_cv = ParsedCV(
        summary="Kế toán viên với 3 năm kinh nghiệm",
        skills=["Excel", "Word", "Kế toán"],
        experience="Công ty ABC - Kế toán viên (2020-2023)",
        education="Đại học Kinh tế - Chuyên ngành Kế toán"
    )
    
    result = analyzer.analyze(sample_cv)
    print(f"Điểm chất lượng: {result['quality_score']:.2f}")
    print(f"Điểm ATS: {analyzer.calculate_ats_score(result)}")
    print(f"Điểm mạnh: {result['strengths']}")
    print(f"Điểm yếu: {result['weaknesses']}") 