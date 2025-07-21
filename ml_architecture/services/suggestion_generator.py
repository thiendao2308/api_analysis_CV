from typing import List, Dict

class SuggestionGenerator:
    """
    BƯỚC 4: Tạo ra các gợi ý "con người hóa" dựa trên kết quả so khớp.
    """

    def generate(self, matched_keywords: List[str], missing_keywords: List[str], 
                 cv_quality: Dict = None, job_category: str = None) -> List[str]:
        """
        BƯỚC 4: Tạo danh sách các gợi ý cải thiện CV.

        Args:
            matched_keywords: Danh sách các kỹ năng đã khớp.
            missing_keywords: Danh sách các kỹ năng còn thiếu.
            cv_quality: Thông tin chất lượng CV.
            job_category: Ngành nghề ứng tuyển.

        Returns:
            Một danh sách các chuỗi gợi ý.
        """
        suggestions = []

        # BƯỚC 4.1: Gợi ý dựa trên các kỹ năng còn thiếu
        if missing_keywords:
            suggestions.append(
                f"💡 Để tăng cơ hội, bạn có thể xem xét bổ sung hoặc tìm hiểu thêm về các kỹ năng sau: {', '.join(missing_keywords[:3])}."
            )
            suggestions.append(
                "📝 Hãy cập nhật những kỹ năng này vào mục 'Kỹ năng' trong CV nếu bạn đã có kinh nghiệm."
            )
        
        # BƯỚC 4.2: Gợi ý dựa trên các kỹ năng đã khớp
        if matched_keywords:
            suggestions.append(
                f"✅ Rất tốt! Bạn đã có những kỹ năng quan trọng mà nhà tuyển dụng tìm kiếm như: {', '.join(matched_keywords[:3])}. Hãy làm nổi bật chúng trong CV và khi phỏng vấn."
            )

        # BƯỚC 4.3: Gợi ý dựa trên chất lượng CV
        if cv_quality:
            quality_score = cv_quality.get('quality_score', 0)
            if quality_score < 0.75:
                suggestions.append("📋 Cải thiện cấu trúc CV với các mục rõ ràng: Tóm tắt, Kinh nghiệm, Kỹ năng, Học vấn.")
            
            if quality_score < 0.5:
                suggestions.append("🎯 Thêm phần tóm tắt/mục tiêu nghề nghiệp để tạo ấn tượng ban đầu tốt.")
            
            if not cv_quality.get('strengths'):
                suggestions.append("💪 Nhấn mạnh các thành tựu và kết quả cụ thể trong kinh nghiệm làm việc.")

        # BƯỚC 4.4: Gợi ý dựa trên ngành nghề
        if job_category:
            industry_suggestions = self._get_industry_specific_suggestions(job_category)
            suggestions.extend(industry_suggestions)

        # BƯỚC 4.5: Gợi ý chung
        if not suggestions:
            suggestions.append("🎉 CV của bạn có vẻ khá phù hợp với yêu cầu. Hãy tự tin ứng tuyển!")
        else:
            suggestions.append("🔧 Hãy tùy chỉnh CV để nhấn mạnh sự phù hợp của bạn với từng vị trí ứng tuyển cụ thể.")

        return suggestions[:5]  # Giới hạn 5 gợi ý

    def _get_industry_specific_suggestions(self, job_category: str) -> List[str]:
        """BƯỚC 4: Tạo gợi ý dựa trên ngành nghề cụ thể"""
        industry_suggestions = {
            "INFORMATION-TECHNOLOGY": [
                "💻 Nhấn mạnh các dự án công nghệ và kỹ năng lập trình cụ thể.",
                "🔧 Thêm thông tin về các công nghệ, framework và tools đã sử dụng.",
                "📊 Liệt kê các metrics về hiệu suất và tối ưu hóa hệ thống."
            ],
            "ENGINEERING": [
                "⚙️ Nhấn mạnh các dự án kỹ thuật và kỹ năng thiết kế.",
                "📐 Thêm thông tin về các phần mềm CAD/CAM và công cụ phân tích.",
                "🏗️ Liệt kê các dự án xây dựng hoặc thiết kế đã tham gia."
            ],
            "FINANCE": [
                "💰 Nhấn mạnh các kỹ năng phân tích tài chính và quản lý ngân sách.",
                "📊 Thêm thông tin về các báo cáo tài chính và phân tích rủi ro.",
                "💼 Liệt kê các chứng chỉ tài chính hoặc kế toán đã có."
            ],
            "SALES": [
                "📈 Nhấn mạnh các thành tích bán hàng và kỹ năng thương lượng.",
                "🤝 Thêm thông tin về việc xây dựng mối quan hệ khách hàng.",
                "🎯 Liệt kê các chỉ tiêu doanh số đã đạt được."
            ],
            "HR": [
                "👥 Nhấn mạnh các kỹ năng quản lý nhân sự và tuyển dụng.",
                "📋 Thêm thông tin về các chính sách nhân sự đã triển khai.",
                "🎓 Liệt kê các chương trình đào tạo và phát triển nhân viên."
            ],
            "MARKETING": [
                "📢 Nhấn mạnh các chiến dịch marketing và kỹ năng sáng tạo.",
                "📱 Thêm thông tin về digital marketing và social media.",
                "📊 Liệt kê các metrics về hiệu quả marketing đã đạt được."
            ]
        }
        
        return industry_suggestions.get(job_category.upper(), [])

    def generate_detailed_suggestions(self, cv_analysis: Dict) -> Dict:
        """
        BƯỚC 4: Tạo gợi ý chi tiết dựa trên phân tích CV toàn diện
        """
        suggestions = {
            'structure': [],
            'content': [],
            'skills': [],
            'formatting': [],
            'overall': []
        }
        
        # Gợi ý về cấu trúc
        if cv_analysis.get('quality_score', 0) < 0.75:
            suggestions['structure'].append("Cải thiện cấu trúc CV với các mục rõ ràng")
            suggestions['structure'].append("Thêm phần tóm tắt/mục tiêu nghề nghiệp")
        
        # Gợi ý về nội dung
        if not cv_analysis.get('parsed_cv', {}).get('experience'):
            suggestions['content'].append("Bổ sung thông tin kinh nghiệm làm việc")
        
        if not cv_analysis.get('parsed_cv', {}).get('education'):
            suggestions['content'].append("Thêm thông tin học vấn và bằng cấp")
        
        # Gợi ý về kỹ năng
        missing_skills = cv_analysis.get('jd_skills', [])
        if missing_skills:
            suggestions['skills'].append(f"Bổ sung các kỹ năng: {', '.join(missing_skills[:3])}")
        
        # Gợi ý về định dạng
        suggestions['formatting'].append("Sử dụng font chữ dễ đọc và khoảng cách hợp lý")
        suggestions['formatting'].append("Đảm bảo CV không quá 2 trang")
        
        # Gợi ý tổng thể
        overall_score = cv_analysis.get('overall_score', 0)
        if overall_score >= 80:
            suggestions['overall'].append("CV của bạn rất tốt! Hãy tự tin ứng tuyển.")
        elif overall_score >= 60:
            suggestions['overall'].append("CV khá tốt, chỉ cần cải thiện một số điểm nhỏ.")
        else:
            suggestions['overall'].append("CV cần được cải thiện đáng kể để tăng cơ hội.")
        
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
    humanized_suggestions = generator.generate(
        matched_keywords=["Python", "Django"],
        missing_keywords=["Docker", "AWS"],
        cv_quality={'quality_score': 0.6},
        job_category="INFORMATION-TECHNOLOGY"
    )

    print("--- BƯỚC 4: Các gợi ý được 'nhân hóa' ---")
    for suggestion in humanized_suggestions:
        print(f"- {suggestion}") 