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
        """Trích xuất job title thông minh từ CV sử dụng LLM"""
        try:
            # Thử dùng LLM trước
            job_title = self._extract_job_title_with_llm(text)
            if job_title:
                return job_title
            
            # Fallback về rules cũ nếu LLM thất bại
            return self._extract_job_title_with_rules(text)
            
        except Exception as e:
            logger.error(f"Error extracting job title: {e}")
            return self._extract_job_title_with_rules(text)

    def _extract_job_title_with_llm(self, text: str) -> Optional[str]:
        """Trích xuất job title bằng LLM"""
        try:
            # Lấy 15 dòng đầu để giảm token usage
            lines = text.split('\n')
            cv_preview = '\n'.join(lines[:15])
            
            prompt = f"""
            Hãy trích xuất job title/vị trí công việc chính từ CV sau đây.
            
            CV TEXT:
            {cv_preview}
            
            Yêu cầu:
            1. Tìm vị trí công việc chính (Frontend Developer, Backend Developer, Full Stack Developer, etc.)
            2. Nếu có nhiều vị trí, chọn vị trí gần nhất hoặc chính
            3. Trả về chỉ tên vị trí, không có giải thích
            
            Trả về: "Job Title" hoặc null nếu không tìm thấy
            """
            
            # Gọi OpenAI API
            import openai
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return None
                
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
            )
            
            job_title = response.choices[0].message.content.strip()
            
            # Clean up response
            if job_title and job_title.lower() != "null" and len(job_title) < 100:
                return job_title
                
            return None
            
        except Exception as e:
            logger.error(f"LLM job title extraction failed: {e}")
            return None

    def _extract_job_title_with_rules(self, text: str) -> Optional[str]:
        """Fallback: trích xuất job title bằng rules cũ"""
        lines = text.split('\n')
        
        # Tìm trong 10 dòng đầu (thường job title ở đầu CV)
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            # Bỏ qua dòng trống hoặc quá ngắn
            if len(line) < 3 or len(line) > 100:
                continue
                
            # Kiểm tra các từ khóa job title đơn giản
            job_keywords = [
                "developer", "engineer", "manager", "specialist", "analyst",
                "designer", "consultant", "advisor", "coordinator", "assistant",
                "director", "supervisor", "lead", "senior", "junior"
            ]
            
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in job_keywords):
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

    def extract_summary(self, text: str) -> Optional[str]:
        """Trích xuất summary/objective từ CV - cải thiện để tự phát hiện tốt hơn"""
        try:
            # Tìm summary trong 30 dòng đầu (tăng từ 20)
            lines = text.split('\n')
            summary_lines = []
            
            # BƯỚC 1: Tìm theo từ khóa section header
            for i, line in enumerate(lines[:30]):
                line = line.strip()
                if not line:
                    continue
                    
                # Tìm các từ khóa summary
                summary_keywords = [
                    'summary', 'objective', 'profile', 'about', 'introduction',
                    'tóm tắt', 'mục tiêu', 'giới thiệu', 'profile', 'overview'
                ]
                
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in summary_keywords):
                    # Lấy 2-4 dòng tiếp theo làm summary
                    for j in range(i+1, min(i+5, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and len(next_line) > 10 and len(next_line) < 300:
                            summary_lines.append(next_line)
                        else:
                            break
                    break
            
            # BƯỚC 2: Nếu không tìm được theo keyword, tự phát hiện summary
            if not summary_lines:
                summary_lines = self._auto_detect_summary(lines)
            
            # BƯỚC 3: Nếu vẫn không có, tạo summary từ job title và skills
            if not summary_lines:
                summary_lines = self._generate_summary_from_context(text)
            
            if summary_lines:
                return ' '.join(summary_lines)
            return None
            
        except Exception as e:
            logger.error(f"Error extracting summary: {e}")
            return None

    def _auto_detect_summary(self, lines: List[str]) -> List[str]:
        """Tự động phát hiện summary dựa trên pattern và nội dung"""
        summary_lines = []
        
        # Tìm trong 10 dòng đầu (thường summary ở đầu CV)
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if not line:
                continue
            
            # Bỏ qua dòng quá ngắn hoặc quá dài
            if len(line) < 15 or len(line) > 250:
                continue
            
            # Bỏ qua dòng có vẻ là header (toàn chữ hoa, có dấu gạch dưới)
            if line.isupper() or '_' in line or line.count('-') > 2:
                continue
            
            # Bỏ qua dòng có vẻ là contact info (email, phone, address)
            if '@' in line or re.search(r'\d{10,}', line) or 'street' in line.lower():
                continue
            
            # Bỏ qua dòng có vẻ là job title (quá ngắn, có từ khóa job)
            job_keywords = ['developer', 'engineer', 'manager', 'specialist', 'analyst']
            if len(line) < 30 and any(keyword in line.lower() for keyword in job_keywords):
                continue
            
            # Kiểm tra xem có phải là câu mô tả không
            if self._is_descriptive_sentence(line):
                summary_lines.append(line)
                if len(summary_lines) >= 3:  # Lấy tối đa 3 dòng
                    break
        
        return summary_lines

    def _is_descriptive_sentence(self, text: str) -> bool:
        """Kiểm tra xem text có phải là câu mô tả không"""
        # Bỏ qua dòng quá ngắn
        if len(text) < 20:
            return False
        
        # Bỏ qua dòng chỉ có danh sách
        if text.count(',') > 3 or text.count('•') > 2:
            return False
        
        # Bỏ qua dòng có vẻ là bullet point
        if text.startswith('•') or text.startswith('-') or text.startswith('*'):
            return False
        
        # Kiểm tra có vẻ là câu mô tả (có động từ, tính từ)
        descriptive_words = [
            'experience', 'skilled', 'proficient', 'expertise', 'background',
            'passionate', 'dedicated', 'motivated', 'creative', 'analytical',
            'kinh nghiệm', 'thành thạo', 'chuyên môn', 'đam mê', 'sáng tạo'
        ]
        
        text_lower = text.lower()
        if any(word in text_lower for word in descriptive_words):
            return True
        
        # Kiểm tra có vẻ là câu hoàn chỉnh (có dấu chấm, dấu phẩy)
        if '.' in text or ',' in text:
            return True
        
        return False

    def _generate_summary_from_context(self, text: str) -> List[str]:
        """Tạo summary từ context của CV (job title, skills, education)"""
        try:
            # Lấy job title
            job_title = self.extract_job_title(text)
            if not job_title:
                job_title = "Software Developer"
            
            # Lấy skills chính
            skills = self.extract_skills(text)
            main_skills = skills[:5] if skills else ["Programming", "Problem Solving"]
            
            # Lấy education
            education = self.extract_education(text)
            degree = "Information Technology" if not education else str(education[0].get('degree', 'Information Technology'))
            
            # Tạo summary tự động
            summary = f"{job_title} với kiến thức về {', '.join(main_skills[:3])}. Tốt nghiệp {degree} và có khả năng học hỏi nhanh."
            
            return [summary]
            
        except Exception as e:
            logger.error(f"Error generating summary from context: {e}")
            return []

    def extract_experience(self, text: str) -> List[Dict]:
        """Trích xuất kinh nghiệm làm việc - cải thiện để nhận diện cấu trúc thực tế"""
        experience = []
        
        try:
            # Tìm experience section
            sections = self.extract_sections(text)
            exp_section = sections.get('experience', '')
            
            if not exp_section:
                # Fallback: tìm trong toàn bộ text
                exp_keywords = ['work experience', 'experience', 'kinh nghiệm', 'công việc']
                for keyword in exp_keywords:
                    if keyword.lower() in text.lower():
                        # Tìm đoạn text sau keyword
                        start_idx = text.lower().find(keyword.lower())
                        if start_idx != -1:
                            # Lấy 500 ký tự sau keyword
                            exp_text = text[start_idx:start_idx + 500]
                            exp_section = exp_text
                            break
            
            if exp_section:
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
                            # Pattern: Company Name | Duration hoặc Company Name - Duration
                            company_duration = re.match(r'(.+?)\s*[|–-]\s*(.+)', second_line)
                            if company_duration:
                                job_info['company'] = company_duration.group(1).strip()
                                job_info['duration'] = company_duration.group(2).strip()
                            else:
                                job_info['company'] = second_line.strip()
                        
                        experience.append(job_info)
            
            # Nếu vẫn không tìm được, tạo experience từ projects
            if not experience:
                projects = self.extract_projects(text)
                if projects:
                    # Chuyển projects thành experience
                    for project in projects:
                        exp_info = {
                            'title': f"Project: {project.get('name', 'Unknown')}",
                            'company': 'Personal Project',
                            'duration': 'Current',
                            'description': project.get('description', '')
                        }
                        experience.append(exp_info)
            
            return experience
            
        except Exception as e:
            logger.error(f"Error extracting experience: {e}")
            return []

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
            summary = self.extract_summary(text)
            
            return {
                "raw_text": text,
                "job_title": job_title,
                "sections": sections,
                "skills": skills,
                "experience": experience,
                "education": education,
                "projects": projects,
                "summary": summary,
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