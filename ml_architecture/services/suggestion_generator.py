from typing import List

class SuggestionGenerator:
    """
    Tạo ra các gợi ý "con người hóa" dựa trên kết quả so khớp.
    """

    def generate(self, matched_keywords: List[str], missing_keywords: List[str]) -> List[str]:
        """
        Tạo danh sách các gợi ý.

        Args:
            matched_keywords: Danh sách các kỹ năng đã khớp.
            missing_keywords: Danh sách các kỹ năng còn thiếu.

        Returns:
            Một danh sách các chuỗi gợi ý.
        """
        suggestions = []

        # Gợi ý dựa trên các kỹ năng còn thiếu
        if missing_keywords:
            suggestions.append(
                f"Để tăng cơ hội, bạn có thể xem xét bổ sung hoặc tìm hiểu thêm về các kỹ năng sau: {', '.join(missing_keywords)}."
            )
            suggestions.append(
                "Hãy cập nhật những kỹ năng này vào mục 'Kỹ năng' trong CV nếu bạn đã có kinh nghiệm."
            )
        
        # Gợi ý dựa trên các kỹ năng đã khớp
        if matched_keywords:
            suggestions.append(
                f"Rất tốt! Bạn đã có những kỹ năng quan trọng mà nhà tuyển dụng tìm kiếm như: {', '.join(matched_keywords)}. Hãy làm nổi bật chúng trong CV và khi phỏng vấn."
            )

        # Gợi ý chung
        if not suggestions:
            suggestions.append("CV của bạn có vẻ khá phù hợp với yêu cầu. Hãy tự tin ứng tuyển!")
        else:
            suggestions.append("Hãy tùy chỉnh CV để nhấn mạnh sự phù hợp của bạn với từng vị trí ứng tuyển cụ thể.")

        return suggestions

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    from cv_parser import CVParser
    
    # Giả lập dữ liệu đã được phân tích
    cv_parser = CVParser()
    sample_cv_text = """
    NGUYỄN VĂN A
    KỸ NĂNG: Python, Django, PostgreSQL
    """
    parsed_cv_data = cv_parser.parse(sample_cv_text)
    
    # Giả lập JD
    parsed_jd_data = {
        "skills": ["Python", "Django", "Docker", "AWS"]
    }

    # Tạo gợi ý
    generator = SuggestionGenerator()
    humanized_suggestions = generator.generate(parsed_jd_data["skills"], [])

    print("--- Các gợi ý được 'nhân hóa' ---")
    for suggestion in humanized_suggestions:
        print(f"- {suggestion}") 