import re
import spacy
from typing import Dict, List, Optional, Tuple
import PyPDF2
import fitz  # PyMuPDF
from docx import Document
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentCVParser:
    def __init__(self):
        """Khởi tạo parser với các pattern và rules thông minh"""
        self.nlp = spacy.load("en_core_web_sm")
        
        # Patterns để nhận diện job title
        self.job_title_patterns = [
            r"(?i)(frontend|front-end|front end)\s+(developer|dev)",
            r"(?i)(backend|back-end|back end)\s+(developer|dev)",
            r"(?i)(fullstack|full-stack|full stack)\s+(developer|dev)",
            r"(?i)(mobile|android|ios)\s+(developer|dev)",
            r"(?i)(data\s+scientist|data\s+analyst)",
            r"(?i)(devops|cloud)\s+(engineer|dev)",
            r"(?i)(qa|quality|test)\s+(engineer|analyst)",
            r"(?i)(ui|ux|user\s+interface)\s+(designer|developer)",
            r"(?i)(system|network)\s+(admin|administrator)",
            r"(?i)(cyber|security)\s+(specialist|engineer)",
            r"(?i)(marketing|digital\s+marketing|seo)\s+(specialist|manager)",
            r"(?i)(sales|business)\s+(representative|manager|director)",
            r"(?i)(hr|human\s+resources)\s+(specialist|manager)",
            r"(?i)(accountant|auditor|financial)\s+(specialist|manager)",
            r"(?i)(teacher|professor|educator)",
            r"(?i)(doctor|nurse|pharmacist|medical)",
            r"(?i)(lawyer|legal|attorney)",
            r"(?i)(consultant|advisor)",
            r"(?i)(designer|graphic|web|product)\s+(designer)",
        ]
        
        # Keywords để nhận diện sections
        self.section_keywords = {
            "experience": [
                "experience", "work experience", "employment", "career", "professional",
                "kinh nghiệm", "kinh nghiệm làm việc", "nghề nghiệp", "công việc"
            ],
            "education": [
                "education", "academic", "degree", "university", "college", "school",
                "học vấn", "bằng cấp", "đại học", "trường học", "tốt nghiệp"
            ],
            "skills": [
                "skills", "technical skills", "competencies", "abilities", "proficiencies",
                "kỹ năng", "kỹ năng kỹ thuật", "năng lực", "khả năng"
            ],
            "projects": [
                "projects", "portfolio", "achievements", "accomplishments",
                "dự án", "danh mục", "thành tựu", "kết quả"
            ],
            "certifications": [
                "certifications", "certificates", "licenses", "accreditations",
                "chứng chỉ", "giấy phép", "chứng nhận"
            ]
        }
        
        # Skills mapping theo ngành nghề
        self.skills_by_category = {
            "INFORMATION-TECHNOLOGY": [
                "python", "java", "javascript", "react", "angular", "vue", "node.js",
                "django", "flask", "spring", "sql", "mongodb", "postgresql", "mysql",
                "aws", "azure", "docker", "kubernetes", "git", "jenkins", "jira",
                "html", "css", "sass", "typescript", "php", "c#", ".net", "ruby",
                "go", "rust", "swift", "kotlin", "android", "ios", "flutter",
                "machine learning", "ai", "data science", "big data", "hadoop",
                "spark", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"
            ],
            "MARKETING": [
                "digital marketing", "seo", "sem", "google ads", "facebook ads",
                "social media", "content marketing", "email marketing", "analytics",
                "google analytics", "facebook pixel", "conversion optimization",
                "brand management", "market research", "customer acquisition",
                "lead generation", "sales funnel", "crm", "hubspot", "mailchimp"
            ],
            "FINANCE": [
                "financial analysis", "accounting", "bookkeeping", "auditing",
                "tax preparation", "budgeting", "forecasting", "investment",
                "risk management", "compliance", "quickbooks", "excel", "sap",
                "oracle", "financial modeling", "valuation", "mergers", "acquisitions"
            ],
            "SALES": [
                "sales", "business development", "account management", "lead generation",
                "prospecting", "negotiation", "closing", "crm", "salesforce",
                "pipeline management", "territory management", "customer relationship",
                "presentation", "communication", "cold calling", "sales strategy"
            ],
            "HR": [
                "recruitment", "talent acquisition", "hiring", "interviewing",
                "onboarding", "employee relations", "performance management",
                "compensation", "benefits", "training", "development", "hr policies",
                "workday", "bamboo hr", "adp", "payroll", "compliance", "diversity"
            ]
        }

    def extract_text_from_file(self, file_path: str) -> str:
        """Trích xuất text từ file CV (PDF, DOCX, TXT)"""
        try:
            if file_path.lower().endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                return self._extract_from_docx(file_path)
            elif file_path.lower().endswith('.txt'):
                return self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        """Trích xuất text từ PDF với fallback methods"""
        text = ""
        
        # Thử PyMuPDF trước (tốt hơn cho layout phức tạp)
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
        
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PyPDF2 failed: {e}")
            return ""

    def _extract_from_docx(self, file_path: str) -> str:
        """Trích xuất text từ DOCX"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {e}")
            return ""

    def _extract_from_txt(self, file_path: str) -> str:
        """Trích xuất text từ TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting from TXT: {e}")
            return ""

    def extract_job_title(self, text: str) -> Optional[str]:
        """Trích xuất job title thông minh từ CV"""
        lines = text.split('\n')
        
        # Tìm trong 10 dòng đầu (thường job title ở đầu CV)
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if not line:
                continue
                
            # Kiểm tra các pattern
            for pattern in self.job_title_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0).strip()
            
            # Kiểm tra các từ khóa job title đơn giản
            job_keywords = [
                "developer", "engineer", "manager", "specialist", "analyst",
                "designer", "consultant", "advisor", "coordinator", "assistant",
                "director", "supervisor", "lead", "senior", "junior"
            ]
            
            words = line.lower().split()
            for word in words:
                if word in job_keywords:
                    return line.strip()
        
        return None

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Trích xuất các section từ CV"""
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Kiểm tra xem line có phải là section header không
            section_found = False
            for section_name, keywords in self.section_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in line.lower():
                        # Lưu section trước đó
                        if current_section and current_content:
                            sections[current_section] = '\n'.join(current_content)
                        
                        # Bắt đầu section mới
                        current_section = section_name
                        current_content = [line]
                        section_found = True
                        break
                if section_found:
                    break
            
            # Nếu không phải header, thêm vào content hiện tại
            if not section_found and current_section:
                current_content.append(line)
        
        # Lưu section cuối cùng
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def extract_skills(self, text: str, job_category: str = None) -> List[str]:
        """Trích xuất skills từ CV"""
        skills = []
        
        # Tìm trong section skills
        sections = self.extract_sections(text)
        skills_section = sections.get('skills', '')
        
        # Tách skills từ skills section
        if skills_section:
            # Tách theo dấu phẩy, chấm phẩy, gạch đầu dòng
            skill_patterns = [
                r'[•\-\*]\s*([^,\n]+)',
                r'([^,\n]+)(?=,|;|\n)',
                r'([^,\n]+)'
            ]
            
            for pattern in skill_patterns:
                matches = re.findall(pattern, skills_section, re.IGNORECASE)
                for match in matches:
                    skill = match.strip()
                    if len(skill) > 2 and skill.lower() not in ['and', 'or', 'the', 'with']:
                        skills.append(skill)
        
        # Tìm skills trong toàn bộ text
        if job_category and job_category in self.skills_by_category:
            category_skills = self.skills_by_category[job_category]
            for skill in category_skills:
                if re.search(rf'\b{re.escape(skill)}\b', text, re.IGNORECASE):
                    if skill not in skills:
                        skills.append(skill)
        
        return list(set(skills))  # Loại bỏ duplicates

    def extract_experience(self, text: str) -> List[Dict]:
        """Trích xuất kinh nghiệm làm việc"""
        experience = []
        sections = self.extract_sections(text)
        exp_section = sections.get('experience', '')
        
        if not exp_section:
            return experience
        
        # Tìm các block kinh nghiệm
        exp_blocks = re.split(r'\n\s*\n', exp_section)
        
        for block in exp_blocks:
            if not block.strip():
                continue
                
            # Tìm job title, company, duration
            lines = block.split('\n')
            if len(lines) >= 2:
                job_info = {
                    'title': lines[0].strip(),
                    'company': '',
                    'duration': '',
                    'description': '\n'.join(lines[1:]).strip()
                }
                
                # Tìm company và duration trong dòng thứ 2
                if len(lines) >= 2:
                    second_line = lines[1]
                    # Pattern: Company Name | Duration
                    company_duration = re.match(r'(.+?)\s*[|–-]\s*(.+)', second_line)
                    if company_duration:
                        job_info['company'] = company_duration.group(1).strip()
                        job_info['duration'] = company_duration.group(2).strip()
                    else:
                        job_info['company'] = second_line.strip()
                
                experience.append(job_info)
        
        return experience

    def extract_education(self, text: str) -> List[Dict]:
        """Trích xuất thông tin học vấn"""
        education = []
        sections = self.extract_sections(text)
        edu_section = sections.get('education', '')
        
        if not edu_section:
            return education
        
        # Tìm các block education
        edu_blocks = re.split(r'\n\s*\n', edu_section)
        
        for block in edu_blocks:
            if not block.strip():
                continue
                
            lines = block.split('\n')
            if len(lines) >= 2:
                edu_info = {
                    'degree': lines[0].strip(),
                    'school': '',
                    'year': '',
                    'description': '\n'.join(lines[1:]).strip()
                }
                
                # Tìm school và year
                if len(lines) >= 2:
                    second_line = lines[1]
                    # Pattern: School Name | Year
                    school_year = re.match(r'(.+?)\s*[|–-]\s*(.+)', second_line)
                    if school_year:
                        edu_info['school'] = school_year.group(1).strip()
                        edu_info['year'] = school_year.group(2).strip()
                    else:
                        edu_info['school'] = second_line.strip()
                
                education.append(edu_info)
        
        return education

    def extract_projects(self, text: str) -> List[Dict]:
        """Trích xuất thông tin dự án"""
        projects = []
        sections = self.extract_sections(text)
        proj_section = sections.get('projects', '')
        
        if not proj_section:
            return projects
        
        # Tìm các block project
        proj_blocks = re.split(r'\n\s*\n', proj_section)
        
        for block in proj_blocks:
            if not block.strip():
                continue
                
            lines = block.split('\n')
            if len(lines) >= 2:
                proj_info = {
                    'name': lines[0].strip(),
                    'description': '\n'.join(lines[1:]).strip()
                }
                projects.append(proj_info)
        
        return projects

    def parse_cv(self, file_path: str, job_category: str = None) -> Dict:
        """Parse CV và trả về thông tin chi tiết"""
        try:
            # Trích xuất text
            text = self.extract_text_from_file(file_path)
            if not text:
                return {"error": "Không thể trích xuất text từ file"}
            
            # Trích xuất các thông tin
            job_title = self.extract_job_title(text)
            sections = self.extract_sections(text)
            skills = self.extract_skills(text, job_category)
            experience = self.extract_experience(text)
            education = self.extract_education(text)
            projects = self.extract_projects(text)
            
            return {
                "raw_text": text,
                "job_title": job_title,
                "sections": sections,
                "skills": skills,
                "experience": experience,
                "education": education,
                "projects": projects,
                "parsed_successfully": True
            }
            
        except Exception as e:
            logger.error(f"Error parsing CV: {str(e)}")
            return {
                "error": f"Lỗi khi parse CV: {str(e)}",
                "parsed_successfully": False
            }

# Tạo instance global
cv_parser = IntelligentCVParser()

def parse_cv_file(file_path: str, job_category: str = None) -> Dict:
    """Wrapper function để parse CV file"""
    return cv_parser.parse_cv(file_path, job_category) 