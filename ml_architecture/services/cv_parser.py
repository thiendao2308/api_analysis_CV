import logging
import re
import spacy
from typing import Dict, List

from ..models.shared_models import ParsedCV

logger = logging.getLogger(__name__)

class CVParser:
    """
    Phân tích một CV dạng văn bản thô để trích xuất các thông tin có cấu trúc như
    tóm tắt, kỹ năng, kinh nghiệm và học vấn.
    """
    def __init__(self, model_name="vi_core_news_lg"):
        """
        Khởi tạo CVParser với mô hình SpaCy.
        Tự động fallback sang mô hình tiếng Anh nếu mô hình tiếng Việt không có sẵn.
        """
        self.nlp = self._load_spacy_model(model_name)
        self.section_keywords = {
            "tóm tắt": r"(mục tiêu nghề nghiệp|giới thiệu|tóm tắt|mục tiêu|summary)",
            "kỹ năng": r"(kỹ năng|chuyên môn|trình độ chuyên môn|skills|technical skills)",
            "kinh nghiệm làm việc": r"(kinh nghiệm làm việc|kinh nghiệm|lịch sử công việc|experience)",
            "học vấn": r"(học vấn|giáo dục|trình độ học vấn|education)",
        }
        self.section_regex = self._compile_section_regex()

    def _compile_section_regex(self) -> re.Pattern:
        """Tạo một biểu thức chính quy để tìm các tiêu đề mục trong CV."""
        all_patterns = "|".join(self.section_keywords.values())
        return re.compile(r"^\s*(" + all_patterns + r")\s*:?\s*$", re.IGNORECASE | re.MULTILINE)

    def _load_spacy_model(self, primary_model: str, fallback_model: str = "en_core_web_sm"):
        """Tải mô hình SpaCy, ưu tiên mô hình chính và fallback nếu cần."""
        try:
            logger.info(f"Đang thử tải mô hình chính: '{primary_model}'...")
            return spacy.load(primary_model)
        except OSError:
            logger.warning(f"Không tìm thấy mô hình chính '{primary_model}'.")
            logger.info(f"Đang thử tải mô hình dự phòng: '{fallback_model}'...")
            try:
                return spacy.load(fallback_model)
            except OSError:
                logger.error(f"Không tìm thấy cả mô hình dự phòng '{fallback_model}'.")
                return None

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Phân chia văn bản CV thành các phần chính dựa trên các tiêu đề."""
        logger.info("Đang phân chia CV thành các mục...")
        sections = {}
        matches = list(self.section_regex.finditer(text))
        last_pos = 0

        for i, match in enumerate(matches):
            section_title_text = match.group(1).lower()
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_content = text[content_start:content_end].strip()

            # Tìm tên mục chuẩn hóa (ví dụ: 'kinh nghiệm làm việc' -> 'kinh nghiệm làm việc')
            for key, pattern in self.section_keywords.items():
                if re.search(pattern, section_title_text, re.IGNORECASE):
                    sections[key] = section_content
                    break
        
        logger.info(f"Các mục đã được nhận diện từ CV: {list(sections.keys())}")
        return sections

    def _extract_skills_from_section(self, section_text: str) -> List[str]:
        """Trích xuất danh sách kỹ năng từ một phần văn bản."""
        logger.info("Đang trích xuất kỹ năng từ mục (logic cải tiến)...")
        text_without_headings = re.sub(r'[^:\n]+:\s*', '', section_text)
        potential_skills = re.split(r'\n|,|&|\s-\s|\s\*\s|\s•\s', text_without_headings)
        
        cleaned_skills = []
        for skill in potential_skills:
            s = skill.strip().rstrip('.').strip()
            if s and len(s) > 1 and len(s.split()) < 5: # Chấp nhận kỹ năng dài hơn một chút
                cleaned_skills.append(s)
        
        logger.info(f"Các kỹ năng sau khi được dọn dẹp: {cleaned_skills}")
        return list(set(cleaned_skills))

    def parse(self, text: str) -> ParsedCV:
        """
        Phân tích toàn bộ văn bản CV và trả về dữ liệu có cấu trúc.
        """
        logger.info("==================================================")
        logger.info("BẮT ĐẦU PHIÊN PHÂN TÍCH CV MỚI")
        logger.info(f"Nội dung CV nhận được (500 ký tự đầu):\n---\n{text[:500]}\n---")
        
        sections = self._extract_sections(text)
        
        skills = []
        if "kỹ năng" in sections:
            skills = self._extract_skills_from_section(sections["kỹ năng"])

        # Sử dụng .get() để an toàn lấy nội dung, mặc định trả về None nếu không có
        summary = sections.get("tóm tắt")
        experience = sections.get("kinh nghiệm làm việc")
        education = sections.get("học vấn")
        
        logger.info(f"Các kỹ năng đã bóc tách được: {skills}")
        logger.info("KẾT THÚC PHIÊN PHÂN TÍCH CV")
        logger.info("==================================================")

        return ParsedCV(
            summary=summary,
            skills=skills,
            experience=experience,
            education=education
        )

    def _load_model(self, model_path: str):
        """
        Tải mô hình SpaCy từ đường dẫn đến file mô hình.
        """
        try:
            logger.info(f"Đang tải mô hình từ đường dẫn: '{model_path}'...")
            return spacy.load(model_path)
        except OSError:
            logger.error(f"Không tìm thấy mô hình từ đường dẫn '{model_path}'.")
            logger.info(f"CVParser sẽ hoạt động ở chế độ cơ bản. Vui lòng chạy 'python -m spacy download {model_path}'")
            return None

# --- Ví dụ sử dụng ---
if __name__ == '__main__':
    # You need to have 'vi_core_news_lg' downloaded
    # python -m spacy download vi_core_news_lg
    parser = CVParser()
    sample_cv = """
    NGUYỄN VĂN A
    Email: nva@email.com | Phone: 0123456789

    MỤC TIÊU NGHỀ NGHIỆP
    Trở thành một lập trình viên Python chuyên nghiệp.

    KINH NGHIỆM LÀM VIỆC
    Công ty ABC (01/2020 - Hiện tại)
    Lập trình viên Python
    - Xây dựng hệ thống backend sử dụng Django.
    - Tối ưu hóa hiệu năng database.

    HỌC VẤN
    Đại học XYZ (2016 - 2020)
    Chuyên ngành: Khoa học máy tính

    KỸ NĂNG
    - Ngôn ngữ: Python, Java
    - Framework: Django, Spring
    - Database: PostgreSQL
    """
    
    parsed_data = parser.parse(sample_cv)
    print("--- Dữ liệu CV đã được bóc tách ---")
    print(parsed_data.model_dump_json(indent=2, ensure_ascii=False))

    # Test với một CV phức tạp hơn
    complex_cv = """
    TRẦN THỊ B
    Email: ttb@gmail.com

    TÓM TẮT
    Software Engineer với 3 năm kinh nghiệm làm việc với Python và các framework web.
    
    KINH NGHIỆM LÀM VIỆC

    Công ty Cổ phần Giải pháp Nhanh (05/2021 - Hiện tại)
    Software Engineer
    - Phát triển các tính năng mới cho sản phẩm X sử dụng Flask.
    - Làm việc với Docker và Kubernetes để triển khai.
    - Tối ưu truy vấn PostgreSQL.

    Công ty Z (01/2020 - 04/2021)
    Junior Developer
    - Hỗ trợ team trong việc maintain code base.
    - Viết unit test.

    HỌC VẤN
    Đại học Bách Khoa TPHCM (2016 - 2020)
    Bằng cấp: Kỹ sư
    Chuyên ngành: Khoa học Máy tính

    KỸ NĂNG
    - Python, JavaScript
    - Flask, React
    - Docker, AWS
    """
    print("\n--- Test với CV phức tạp hơn ---")
    parsed_complex_cv = parser.parse(complex_cv)
    print(parsed_complex_cv.model_dump_json(indent=2, ensure_ascii=False)) 