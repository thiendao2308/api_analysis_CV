# Các section chính trong CV
CV_SECTIONS = {
    "contact": ["contact", "liên hệ", "thông tin liên hệ", "thông tin cá nhân", "thông tin", "infomation", "contact infomation"],
    "experience": ["experience", "kinh nghiệm", "work experience", "kinh nghiệm làm việc"],
    "skills": ["skills", "kĩ năng", "các kĩ năng", "kĩ năng chuyên ngành", "kĩ năng chuyên môn", "technical skills", "soft skills", "languages", "tools", "frameworks"],
    "education": ["education", "học vấn", "trình độ học vấn", "degree", "university", "gpa", "graduation", "courses"],
    "projects": ["projects", "dự án", "project", "description", "technologies", "role", "duration", "results"]
}

# Từ khóa cho từng section
SECTION_KEYWORDS = {
    "contact": ["email", "phone", "address", "linkedin", "github"],
    "experience": ["years", "responsibilities", "achievements", "projects"],
    "skills": ["technical", "soft skills", "languages", "tools", "frameworks"],
    "education": ["degree", "university", "gpa", "graduation", "courses"],
    "projects": ["description", "technologies", "role", "duration", "results"]
}

# Các định dạng file được chấp nhận
ALLOWED_FILE_TYPES = [".pdf", ".docx"]

# Cấu hình phân tích
ANALYSIS_CONFIG = {
    "min_section_length": 50,  # Số từ tối thiểu cho mỗi section
    "max_keyword_count": 5,    # Số lần xuất hiện tối đa của một từ khóa
    "min_overall_score": 0.6,  # Điểm tối thiểu để CV được coi là tốt
}

# Các vấn đề về định dạng
FORMAT_ISSUES = {
    "too_short": "CV quá ngắn (dưới 200 từ)",
    "missing_sections": "Thiếu các section chính",
    "all_caps": "Có đoạn text viết hoa quá dài",
    "inconsistent_format": "Định dạng không nhất quán"
}

# Gợi ý mặc định
DEFAULT_SUGGESTIONS = {
    "format": [
        "Sử dụng font chữ và kích thước nhất quán",
        "Căn lề và khoảng cách hợp lý",
        "Sử dụng bullet points cho các danh sách"
    ],
    "content": [
        "Thêm số liệu cụ thể cho các thành tích",
        "Mô tả chi tiết các dự án",
        "Liệt kê đầy đủ các kỹ năng kỹ thuật"
    ]
} 