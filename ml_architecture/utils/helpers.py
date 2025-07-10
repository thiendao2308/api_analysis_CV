import re
import spacy
from typing import List, Dict, Any, Tuple
from .constants import ANALYSIS_CONFIG, FORMAT_ISSUES

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def clean_text(text: str) -> str:
    """
    Làm sạch text: loại bỏ ký tự đặc biệt, chuẩn hóa khoảng trắng
    """
    if not text:
        return ""
    
    # Loại bỏ ký tự đặc biệt
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Tính độ tương đồng giữa hai đoạn text sử dụng spaCy
    """
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    return doc1.similarity(doc2)

def extract_keywords(text: str) -> List[str]:
    """
    Trích xuất từ khóa từ text sử dụng spaCy
    """
    doc = nlp(text)
    keywords = []
    
    # Lấy các từ khóa là danh từ, động từ, tính từ
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and len(token.text) > 2:
            keywords.append(token.text.lower())
    
    return list(set(keywords))

def find_section_boundaries(text: str, section_keywords: List[str]) -> Tuple[int, int]:
    """
    Tìm vị trí bắt đầu và kết thúc của một section
    """
    lines = text.split('\n')
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        line = line.lower()
        if any(keyword.lower() in line for keyword in section_keywords):
            if start_idx == -1:
                start_idx = i
            else:
                end_idx = i
                break
    
    if start_idx != -1 and end_idx == -1:
        end_idx = len(lines)
    
    return start_idx, end_idx

def count_keyword_occurrences(text: str, keyword: str) -> int:
    """
    Đếm số lần xuất hiện của một từ khóa trong text
    """
    return text.lower().count(keyword.lower())

def calculate_section_score(section_text: str, keywords: List[str]) -> float:
    """
    Tính điểm cho một section dựa trên số từ khóa xuất hiện
    """
    if not section_text or not keywords:
        return 0.0
    
    matches = sum(1 for keyword in keywords if keyword.lower() in section_text.lower())
    return min(1.0, matches / len(keywords))

def generate_section_suggestions(section_text: str, section_name: str, keywords: List[str]) -> List[str]:
    """
    Tạo gợi ý cải thiện cho một section
    """
    suggestions = []
    
    # Kiểm tra độ dài
    if len(section_text.split()) < 50:
        suggestions.append(f"Bổ sung thêm nội dung cho section {section_name}")
    
    # Kiểm tra từ khóa
    missing_keywords = [k for k in keywords if k.lower() not in section_text.lower()]
    if missing_keywords:
        suggestions.append(f"Thêm các từ khóa: {', '.join(missing_keywords[:3])}")
    
    return suggestions

def format_score(score: float) -> str:
    """
    Định dạng điểm số thành phần trăm
    """
    return f"{score * 100:.1f}%"

def get_section_score(section_content: str, keywords: List[str]) -> float:
    """
    Tính điểm cho một section dựa trên số từ khóa xuất hiện
    """
    if not section_content or not keywords:
        return 0.0
        
    # Đếm số từ khóa xuất hiện
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in section_content.lower())
    
    # Tính điểm dựa trên tỷ lệ từ khóa xuất hiện
    return min(1.0, keyword_count / len(keywords))

def validate_cv_format(cv_content: str) -> List[str]:
    """
    Kiểm tra định dạng CV và trả về danh sách lỗi
    """
    issues = []
    
    # Kiểm tra độ dài
    if len(cv_content.split()) < ANALYSIS_CONFIG["min_cv_length"]:
        issues.append(FORMAT_ISSUES["length"])
        
    # Kiểm tra cấu trúc
    if not any(section in cv_content.lower() for section in ["experience", "skills", "education"]):
        issues.append(FORMAT_ISSUES["structure"])
        
    # Kiểm tra viết hoa
    if re.search(r'[A-Z]{3,}', cv_content):
        issues.append(FORMAT_ISSUES["capitalization"])
        
    # Kiểm tra khoảng cách
    if re.search(r'\n{3,}', cv_content):
        issues.append(FORMAT_ISSUES["spacing"])
        
    return issues

def format_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Định dạng kết quả phân tích để hiển thị
    """
    formatted_result = result.copy()
    
    # Định dạng điểm số
    formatted_result["overall_score"] = format_score(result["overall_score"])
    
    # Định dạng điểm section
    for section in formatted_result["section_analysis"].values():
        section["score"] = format_score(section["score"])
        
    return formatted_result

def get_improvement_priority(issues: List[str]) -> int:
    """
    Xác định mức độ ưu tiên của các vấn đề cần cải thiện
    """
    priority_map = {
        FORMAT_ISSUES["structure"]: 1,
        FORMAT_ISSUES["length"]: 2,
        FORMAT_ISSUES["capitalization"]: 3,
        FORMAT_ISSUES["spacing"]: 4,
        FORMAT_ISSUES["font"]: 5
    }
    
    return min(priority_map.get(issue, 6) for issue in issues) 