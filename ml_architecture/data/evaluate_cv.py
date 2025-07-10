import os
import json
import re
from collections import Counter

# Bộ từ khóa section (giống như trong script tách section)
SECTION_KEYWORDS = [
    r"(?i)education|học vấn|trình độ học vấn|academic|academic background|học tập|bằng cấp|degree|qualification|trình độ",
    r"(?i)experience|kinh nghiệm|quá trình làm việc|work history|work experience|lịch sử công việc|employment history|professional experience|career history",
    r"(?i)skills|kỹ năng|technical skills|soft skills|professional skills|competencies|abilities|proficiencies|expertise|chuyên môn",
    r"(?i)certificates?|chứng chỉ|certifications?|licenses?|diplomas?|awards?|achievements?|recognition|giải thưởng",
    r"(?i)projects?|dự án|project experience|project history|key projects|major projects|project portfolio",
    r"(?i)awards?|giải thưởng|honors?|recognition|achievements?|accomplishments?|merits?",
    r"(?i)activities|hoạt động|volunteer|volunteering|community service|extracurricular|ngoại khóa|social activities",
    r"(?i)contact|liên hệ|thông tin liên lạc|contact information|personal details|thông tin cá nhân|address|địa chỉ|phone|số điện thoại|email|email address",
    r"(?i)summary|tóm tắt|giới thiệu bản thân|profile|personal summary|career summary|professional summary|overview|introduction",
    r"(?i)languages?|ngoại ngữ|language skills|foreign languages?|language proficiency|language abilities",
    r"(?i)interests?|sở thích|hobbies|personal interests?|leisure activities|recreational activities",
    r"(?i)references?|người tham chiếu|referees?|character references?|professional references?",
    r"(?i)personal information|thông tin cá nhân|personal details|personal data|background|personal background",
    r"(?i)objective|mục tiêu|career objective|professional objective|goals?|career goals?|aspirations?",
    r"(?i)work experience|kinh nghiệm làm việc|employment|job history|professional background|career experience",
    r"(?i)technical skills|kỹ năng kỹ thuật|technical expertise|technical competencies|technical abilities|technical knowledge",
    r"(?i)soft skills|kỹ năng mềm|interpersonal skills|communication skills|leadership skills|teamwork skills",
    r"(?i)computer skills|kỹ năng máy tính|it skills|digital skills|software skills|programming skills|coding skills",
    r"(?i)leadership|lãnh đạo|management|quản lý|supervision|team leadership|project leadership",
    r"(?i)research|nghiên cứu|research experience|research projects?|academic research|scientific research",
    r"(?i)publications?|công bố|papers?|articles?|journals?|conferences?|presentations?",
    r"(?i)training|đào tạo|courses?|workshops?|seminars?|professional development|continuing education",
    r"(?i)internships?|thực tập|internship experience|practical training|field experience|practical work",
    r"(?i)achievements?|thành tựu|accomplishments?|successes?|milestones?|key achievements?|major accomplishments?",
    r"(?i)responsibilities?|trách nhiệm|duties?|roles?|functions?|job responsibilities?|work duties?",
    r"(?i)technologies?|công nghệ|tools?|software|platforms?|frameworks?|languages?|programming languages?",
    r"(?i)industries?|ngành|sectors?|fields?|domains?|business areas?|industry experience",
    r"(?i)companies?|công ty|organizations?|employers?|workplaces?|companies worked for|previous employers?",
    r"(?i)positions?|chức vụ|job titles?|roles?|designations?|job positions?|work roles?",
    r"(?i)education details|chi tiết học vấn|academic qualifications?|educational background|study history|academic history",
    r"(?i)work details|chi tiết công việc|job details|employment details|work information|job information",
    r"(?i)skills details|chi tiết kỹ năng|skill information|competency details|expertise details|proficiency details"
]

SECTION_PATTERN = re.compile(r"^\s*({})\s*[:：]?\s*$".format('|'.join([k.replace('(?i)','') for k in SECTION_KEYWORDS])), re.IGNORECASE)

# Danh sách stopwords
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'i', 'you', 'your', 'yours', 'yourself', 'yourselves', 'we', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'shall', 'ought', 'need', 'dare',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'also', 'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
    'if', 'else', 'unless', 'until', 'while', 'because', 'since', 'although', 'though', 'even', 'whether', 'either', 'neither',
    'but', 'however', 'nevertheless', 'nonetheless', 'still', 'yet', 'though', 'although', 'even', 'though',
    'or', 'nor', 'either', 'neither', 'both', 'and', 'not', 'only', 'but', 'also', 'as', 'well', 'as',
    'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}

CRITERIA_FILE = os.path.join(os.path.dirname(__file__), 'cv_criteria_final.json')

def extract_sections_from_text(text):
    """Tách section từ text CV"""
    lines = text.split('\n') if '\n' in text else text.split('.')
    sections = {}
    current_section = 'Other'
    sections[current_section] = []
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        match = SECTION_PATTERN.match(line_strip)
        if match:
            current_section = match.group(1).strip()
            if current_section not in sections:
                sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(line_strip)
    
    return {k: v for k, v in sections.items() if v}

def extract_entities_from_sections(sections):
    """Trích xuất entity từ các section"""
    entities = {}
    for section, lines in sections.items():
        section_entities = []
        for line in lines:
            words = re.findall(r'\b\w{3,}\b', line.lower())
            for word in words:
                if word not in STOPWORDS and len(word) >= 3:
                    section_entities.append(word)
        entities[section] = section_entities
    return entities

def evaluate_cv(cv_text, job_category):
    """Đánh giá CV dựa trên bộ tiêu chí"""
    
    # Đọc bộ tiêu chí
    with open(CRITERIA_FILE, 'r', encoding='utf-8') as f:
        criteria = json.load(f)
    
    if job_category not in criteria:
        return {
            'error': f'Không tìm thấy tiêu chí cho ngành nghề: {job_category}'
        }
    
    job_criteria = criteria[job_category]
    
    # Tách section và entity từ CV
    sections = extract_sections_from_text(cv_text)
    entities = extract_entities_from_sections(sections)
    
    # Chỉ đánh giá entity, không đánh giá section vì thống kê chưa ổn
    required_entities = job_criteria.get('required_entities', {})
    optional_entities = job_criteria.get('optional_entities', {})
    
    entity_score = 0
    entity_feedback = []
    found_entities_summary = []
    
    # Đánh giá entity trong tất cả các section có sẵn
    all_entities = []
    for section_entities in entities.values():
        all_entities.extend(section_entities)
    
    # Kiểm tra entity phù hợp với ngành nghề
    for section, required_entity_list in required_entities.items():
        if required_entity_list:
            found_entities = [e for e in required_entity_list if e in all_entities]
            if found_entities:
                entity_score += len(found_entities) / len(required_entity_list)
                found_entities_summary.extend(found_entities)
                entity_feedback.append(f"✅ Tìm thấy kỹ năng phù hợp: {', '.join(found_entities)}")
    
    # Tính điểm tổng dựa trên entity
    total_score = entity_score / len(required_entities) if required_entities else 0
    
    # Đánh giá mức độ
    if total_score >= 0.8:
        level = "Xuất sắc"
        overall_feedback = "CV của bạn rất phù hợp với ngành nghề này!"
    elif total_score >= 0.6:
        level = "Tốt"
        overall_feedback = "CV của bạn khá phù hợp với ngành nghề này."
    elif total_score >= 0.4:
        level = "Trung bình"
        overall_feedback = "CV của bạn cần bổ sung thêm kỹ năng chuyên môn."
    else:
        level = "Cần cải thiện"
        overall_feedback = "CV của bạn cần được cải thiện để phù hợp hơn với ngành nghề này."
    
    # Thêm gợi ý cải thiện
    improvement_suggestions = []
    if total_score < 0.8:
        improvement_suggestions.append("Hãy bổ sung thêm các kỹ năng chuyên môn liên quan đến ngành nghề.")
        improvement_suggestions.append("Mô tả chi tiết hơn về kinh nghiệm làm việc.")
        improvement_suggestions.append("Thêm các chứng chỉ hoặc bằng cấp liên quan.")
    
    return {
        'job_category': job_category,
        'total_score': round(total_score * 100, 2),
        'level': level,
        'overall_feedback': overall_feedback,
        'entity_feedback': entity_feedback,
        'found_entities': list(set(found_entities_summary)),  # Loại bỏ trùng lặp
        'improvement_suggestions': improvement_suggestions
    }

# Test với một CV mẫu
if __name__ == "__main__":
    # CV mẫu cho ngành ACCOUNTANT
    sample_cv = """
    EDUCATION
    Bachelor of Accounting, University of Economics
    
    EXPERIENCE
    Accountant at ABC Company
    - Prepared financial statements
    - Managed accounts payable and receivable
    - Performed monthly reconciliations
    
    SKILLS
    - Financial reporting
    - Accounting software
    - Tax preparation
    - Microsoft Excel
    """
    
    result = evaluate_cv(sample_cv, "ACCOUNTANT")
    print(json.dumps(result, ensure_ascii=False, indent=2)) 