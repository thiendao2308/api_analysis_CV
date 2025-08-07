import openai
import os
import logging
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from dotenv import load_dotenv
import re

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

class IntelligentJDMatcher:
    """Intelligent JD matching sử dụng LLM với family mapping và skill normalization"""
    
    def __init__(self):
        self.client = None
        if OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("✅ Intelligent JD Matcher initialized with OpenAI")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
        
        # Skill family mappings
        self.skill_families = {
            "react": ["react", "reactjs", "react.js", "react native"],
            "vue": ["vue", "vue.js", "vuejs", "nuxt"],
            "angular": ["angular", "angularjs", "angular.js"],
            "javascript": ["javascript", "js", "ecmascript", "es6", "es7"],
            "typescript": ["typescript", "ts"],
            "node": ["node", "node.js", "nodejs", "express", "express.js"],
            "dotnet": [".net", "dotnet", "asp.net", "asp.net core", "asp.net mvc", ".net framework", ".net core", "c#"],
            "sql": ["sql", "mysql", "postgresql", "sql server", "ms sql server", "oracle", "sqlite"],
            "mongodb": ["mongodb", "mongo", "nosql"],
            "git": ["git", "github", "gitlab", "bitbucket", "version control"],
            "docker": ["docker", "container", "kubernetes", "k8s"],
            "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
            "azure": ["azure", "microsoft azure", "cloud"],
            "agile": ["agile", "scrum", "kanban", "sprint", "sprint planning"],
            "devops": ["devops", "ci/cd", "jenkins", "gitlab ci", "github actions"],
            "api": ["api", "rest api", "graphql", "api integration", "web api"],
            "html": ["html", "html5"],
            "css": ["css", "css3", "sass", "scss", "less", "bootstrap", "tailwind"],
            "php": ["php", "laravel", "codeigniter", "wordpress"],
            "python": ["python", "django", "flask", "fastapi"],
            "java": ["java", "spring", "spring boot", "maven", "gradle"],
            "mobile": ["react native", "flutter", "ionic", "xamarin", "android", "ios"],
            "testing": ["jest", "mocha", "cypress", "selenium", "unit testing", "integration testing"],
            "ui_ux": ["ui", "ux", "user interface", "user experience", "figma", "sketch", "adobe xd"],
            "seo": ["seo", "search engine optimization", "sem", "ppc"],
            "marketing": ["digital marketing", "social media", "content marketing", "email marketing"],
            "finance": ["financial modeling", "excel", "quickbooks", "sap", "oracle"],
            "hr": ["recruitment", "talent acquisition", "hris", "hr software", "employee relations"],
            "sales": ["crm", "salesforce", "lead generation", "customer relationship"],
            "healthcare": ["emr", "electronic medical records", "patient care", "clinical"],
            "education": ["lms", "learning management system", "e-learning", "curriculum development"],
            # Thêm skill families cho design và video editing
            "design_tools": ["canva", "figma", "sketch", "adobe xd", "photoshop", "illustrator", "indesign", "design", "thiết kế"],
            "video_editing": ["capcut", "premiere pro", "after effects", "final cut pro", "davinci resolve", "video editing", "edit video", "video production"],
            "social_media": ["tiktok", "instagram", "facebook", "twitter", "linkedin", "youtube", "social media", "social media management"],
            "content_creation": ["content creation", "content marketing", "copywriting", "blogging", "video content", "photo editing"],
            "office_tools": ["microsoft office", "word", "excel", "powerpoint", "google docs", "google sheets", "office"],
            "communication": ["communication", "giao tiếp", "presentation", "public speaking", "writing", "tiếng anh"]
        }
    
    def _normalize_skill(self, skill: str) -> str:
        """Chuẩn hóa skill name"""
        skill_lower = skill.lower().strip()
        
        # Remove common suffixes/prefixes
        skill_lower = re.sub(r'\s*\([^)]*\)', '', skill_lower)  # Remove (description)
        skill_lower = re.sub(r'\s*\[[^\]]*\]', '', skill_lower)  # Remove [version]
        
        # Normalize common variations
        skill_lower = skill_lower.replace('javascript', 'js')
        skill_lower = skill_lower.replace('reactjs', 'react')
        skill_lower = skill_lower.replace('vuejs', 'vue')
        skill_lower = skill_lower.replace('angularjs', 'angular')
        skill_lower = skill_lower.replace('nodejs', 'node')
        skill_lower = skill_lower.replace('expressjs', 'express')
        skill_lower = skill_lower.replace('asp.net', 'dotnet')
        skill_lower = skill_lower.replace('.net', 'dotnet')
        skill_lower = skill_lower.replace('sql server', 'sql')
        skill_lower = skill_lower.replace('ms sql server', 'sql')
        
        # Normalize design and video editing tools
        skill_lower = skill_lower.replace('edit video', 'video editing')
        skill_lower = skill_lower.replace('thiết kế cơ bản', 'design')
        skill_lower = skill_lower.replace('thiết kế', 'design')
        skill_lower = skill_lower.replace('giao tiếp', 'communication')
        
        # Remove extra spaces
        skill_lower = re.sub(r'\s+', ' ', skill_lower).strip()
        
        return skill_lower
    
    def _find_skill_family_matches(self, cv_skills: List[str], jd_skills: List[str]) -> List[Tuple[str, str]]:
        """Tìm matches dựa trên skill families với logic tránh duplicate"""
        family_matches = []
        matched_jd_skills = set()  # Track JD skills đã được match
        
        cv_normalized = [self._normalize_skill(skill) for skill in cv_skills]
        jd_normalized = [self._normalize_skill(skill) for skill in jd_skills]
        
        for family_name, family_skills in self.skill_families.items():
            cv_family_matches = []
            jd_family_matches = []
            
            # Find CV skills in this family
            for i, cv_skill in enumerate(cv_normalized):
                if any(fs in cv_skill for fs in family_skills):
                    cv_family_matches.append(cv_skills[i])
            
            # Find JD skills in this family (only unmatched ones)
            for i, jd_skill in enumerate(jd_normalized):
                if any(fs in jd_skill for fs in family_skills) and jd_skills[i] not in matched_jd_skills:
                    jd_family_matches.append(jd_skills[i])
            
            # Match CV skills to JD skills (one-to-one mapping)
            if cv_family_matches and jd_family_matches:
                # Sort by priority (exact matches first, then partial)
                for cv_skill in cv_family_matches:
                    for jd_skill in jd_family_matches:
                        if jd_skill not in matched_jd_skills:
                            family_matches.append((cv_skill, jd_skill))
                            matched_jd_skills.add(jd_skill)
                            break  # Move to next CV skill
        
        return family_matches
    
    def intelligent_matching(self, cv_skills: List[str], jd_skills: List[str]) -> Dict:
        """Intelligent matching với family mapping và skill normalization"""
        logger.info(f"Starting intelligent JD matching: CV={len(cv_skills)} skills, JD={len(jd_skills)} skills")
        
        # 1. Exact matches (case-insensitive)
        exact_matches = self._find_exact_matches(cv_skills, jd_skills)
        
        # 2. Family-based matches
        family_matches = self._find_skill_family_matches(cv_skills, jd_skills)
        family_matched_cv = [match[0] for match in family_matches]
        family_matched_jd = [match[1] for match in family_matches]
        
        # 3. Semantic matches (LLM-based) for remaining skills
        exclude_cv = exact_matches + family_matched_cv
        exclude_jd = family_matched_jd
        semantic_matches = self._find_semantic_matches(cv_skills, jd_skills, exclude_cv, exclude_jd)
        
        # 4. Combine all matches
        all_matches = exact_matches + family_matched_cv + semantic_matches
        missing_skills = self._find_missing_skills(jd_skills, all_matches)
        
        # 5. Calculate match score
        match_score = self._calculate_match_score(len(all_matches), len(jd_skills))
        
        return {
            "matching_skills": all_matches,
            "missing_skills": missing_skills,
            "exact_matches": exact_matches,
            "family_matches": family_matched_cv,
            "semantic_matches": semantic_matches,
            "match_score": match_score,
            "total_cv_skills": len(cv_skills),
            "total_jd_skills": len(jd_skills),
            "family_mapping_details": family_matches
        }
    
    def _find_exact_matches(self, cv_skills: List[str], jd_skills: List[str]) -> List[str]:
        """Tìm exact matches với case-insensitive và xử lý dấu ngoặc"""
        cv_skills_lower = [self._normalize_skill(skill) for skill in cv_skills]
        jd_skills_lower = [self._normalize_skill(skill) for skill in jd_skills]
        
        exact_matches = []
        for cv_skill, cv_lower in zip(cv_skills, cv_skills_lower):
            # Kiểm tra exact match
            if cv_lower in jd_skills_lower:
                exact_matches.append(cv_skill)
                continue
            
            # Kiểm tra partial match (trường hợp có dấu ngoặc)
            for jd_skill, jd_lower in zip(jd_skills, jd_skills_lower):
                # Loại bỏ dấu ngoặc và so sánh
                cv_clean = re.sub(r'\s*\([^)]*\)', '', cv_lower).strip()
                jd_clean = re.sub(r'\s*\([^)]*\)', '', jd_lower).strip()
                
                if cv_clean == jd_clean and cv_clean:
                    exact_matches.append(cv_skill)
                    break
        
        logger.info(f"Found {len(exact_matches)} exact matches")
        return exact_matches
    
    def _find_semantic_matches(self, cv_skills: List[str], jd_skills: List[str], exclude_cv: List[str], exclude_jd: List[str]) -> List[str]:
        """Tìm semantic matches với enhanced prompt tối ưu cho từng ngành"""
        if not self.client:
            logger.warning("OpenAI client not available, skipping semantic matching")
            return []
        
        semantic_matches = []
        exclude_cv_lower = [self._normalize_skill(skill) for skill in exclude_cv]
        exclude_jd_lower = [self._normalize_skill(skill) for skill in exclude_jd]
        
        # Group skills for batch processing
        cv_remaining = [skill for skill in cv_skills if self._normalize_skill(skill) not in exclude_cv_lower]
        jd_remaining = [skill for skill in jd_skills if self._normalize_skill(skill) not in exclude_jd_lower]
        
        if not cv_remaining or not jd_remaining:
            return semantic_matches
        
        try:
            # Tạo prompt tối ưu cho từng ngành nghề
            industry_specific_prompt = self._get_industry_specific_prompt(cv_remaining, jd_remaining)
            
            prompt = f"""
Bạn là chuyên gia phân tích skills matching với 15+ năm kinh nghiệm trong nhiều ngành nghề. 
Hãy phân tích semantic similarity giữa skills CV và JD một cách chính xác và chi tiết.

SKILLS CV: {', '.join(cv_remaining)}
SKILLS JD: {', '.join(jd_remaining)}

{industry_specific_prompt}

NGUYÊN TẮC MATCHING CHI TIẾT:

1. **EXACT MATCHES**: Tìm các skills có nghĩa giống hệt nhau
   - "JavaScript" ↔ "JS", "ReactJS" ↔ "React", "ASP.NET" ↔ ".NET"
   - "Canva" ↔ "Graphic Design (Canva)", "Capcut" ↔ "Video Editing (Capcut)"

2. **FUNCTIONAL EQUIVALENTS**: Tìm các skills có chức năng tương đương
   - "Microsoft Office" ↔ "Excel, Word, PowerPoint"
   - "Digital Marketing" ↔ "Social Media Marketing, Content Marketing"
   - "Problem Solving" ↔ "Analytical Thinking, Critical Thinking"

3. **SKILL LEVEL MATCHING**: Phân biệt basic/intermediate/advanced
   - "Basic Design" ↔ "Canva, Photoshop", "Advanced Design" ↔ "Illustrator, InDesign"
   - "Basic Programming" ↔ "HTML, CSS", "Advanced Programming" ↔ "JavaScript, Python"

4. **INDUSTRY CONTEXT**: Hiểu context ngành nghề cụ thể
   - IT: "Git" ↔ "Version Control", "API" ↔ "REST API"
   - Marketing: "SEO" ↔ "Search Engine Optimization", "PPC" ↔ "Google Ads"
   - Finance: "Excel" ↔ "Financial Modeling", "QuickBooks" ↔ "Accounting"

5. **TOOL EQUIVALENTS**: Các công cụ cùng loại
   - "Figma" ↔ "Sketch, Adobe XD" (UI/UX tools)
   - "Premiere Pro" ↔ "Capcut, After Effects" (Video editing)
   - "Salesforce" ↔ "HubSpot, Pipedrive" (CRM tools)

6. **LANGUAGE VARIATIONS**: Xử lý tiếng Việt/Anh
   - "Thiết kế" ↔ "Design", "Giao tiếp" ↔ "Communication"
   - "Lập trình" ↔ "Programming", "Phân tích" ↔ "Analysis"

7. **COMPOUND SKILLS**: Skills phức hợp
   - "Full-stack Development" ↔ "Frontend + Backend skills"
   - "Digital Marketing" ↔ "Social Media + Content + SEO"
   - "Data Analysis" ↔ "Excel + SQL + Statistics"

LOẠI TRỪ:
- Skills quá chung chung (Management, Leadership)
- Skills không liên quan đến ngành nghề
- Skills có level quá khác biệt (Basic vs Expert)

TRẢ VỀ: Chỉ tên skill từ CV có thể match, mỗi skill một dòng, format: "Skill Name"
Ví dụ:
"JavaScript"
"Microsoft Office"
"Digital Marketing"
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Improved parsing logic
            matched_skills = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip comments/headers
                    # Remove quotes if present
                    skill_name = line.strip('"').strip("'").strip()
                    if skill_name and skill_name in cv_remaining:
                        matched_skills.append(skill_name)
            
            # Additional parsing for different formats
            additional_matches = []
            
            # Pattern 1: "Skill Name" (quoted)
            quoted_pattern = r'"([^"]+)"'
            quoted_matches = re.findall(quoted_pattern, content)
            for match in quoted_matches:
                if match in cv_remaining and match not in matched_skills:
                    additional_matches.append(match)
            
            # Pattern 2: "- Skill Name" (bullet points)
            bullet_pattern = r'-\s*([^\n]+)'
            bullet_matches = re.findall(bullet_pattern, content)
            for match in bullet_matches:
                skill = match.strip().split()[0]  # Take first word as skill
                if skill in cv_remaining and skill not in matched_skills:
                    additional_matches.append(skill)
            
            # Pattern 3: "Skill Name match" (explicit matching)
            match_pattern = r'([A-Za-z\s\.]+)\s+match'
            explicit_matches = re.findall(match_pattern, content)
            for match in explicit_matches:
                skill = match.strip()
                if skill in cv_remaining and skill not in matched_skills:
                    additional_matches.append(skill)
            
            # Combine all matches
            matched_skills.extend(additional_matches)
            
            # Remove duplicates while preserving order
            seen = set()
            semantic_matches = []
            for skill in matched_skills:
                if skill not in seen:
                    seen.add(skill)
                    semantic_matches.append(skill)
            
            logger.info(f"Found {len(semantic_matches)} semantic matches")
            return semantic_matches
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return []
    
    def _get_industry_specific_prompt(self, cv_skills: List[str], jd_skills: List[str]) -> str:
        """Tạo prompt tối ưu cho từng ngành nghề cụ thể"""
        
        # Phân tích ngành nghề dựa trên skills
        industry_keywords = {
            "IT": ["programming", "coding", "development", "software", "web", "app", "database", "api", "git", "framework"],
            "Marketing": ["marketing", "social media", "content", "seo", "ads", "campaign", "brand", "digital", "analytics"],
            "Finance": ["finance", "accounting", "excel", "modeling", "analysis", "budget", "financial", "audit", "tax"],
            "HR": ["hr", "recruitment", "hiring", "talent", "employee", "personnel", "training", "compensation"],
            "Sales": ["sales", "crm", "customer", "lead", "pipeline", "negotiation", "business", "account"],
            "Design": ["design", "graphic", "ui", "ux", "visual", "creative", "art", "illustration", "branding"],
            "Healthcare": ["healthcare", "medical", "patient", "clinical", "nursing", "pharmacy", "hospital", "care"],
            "Education": ["education", "teaching", "learning", "curriculum", "training", "academic", "student", "course"]
        }
        
        # Xác định ngành nghề dựa trên skills
        detected_industry = "General"
        max_score = 0
        
        for industry, keywords in industry_keywords.items():
            score = 0
            all_skills = ' '.join(cv_skills + jd_skills).lower()
            for keyword in keywords:
                if keyword in all_skills:
                    score += 1
            if score > max_score:
                max_score = score
                detected_industry = industry
        
        # Tạo prompt tối ưu cho ngành nghề cụ thể
        industry_prompts = {
            "IT": """
NGÀNH CÔNG NGHỆ THÔNG TIN:
- Programming Languages: "JavaScript" ↔ "JS", "Python" ↔ "Python", "C#" ↔ ".NET"
- Frameworks: "React" ↔ "ReactJS", "Vue.js" ↔ "Vue", "ASP.NET" ↔ ".NET Core"
- Tools: "Git" ↔ "Version Control", "Postman" ↔ "API Testing", "Docker" ↔ "Containerization"
- Databases: "SQL" ↔ "MySQL, PostgreSQL", "MongoDB" ↔ "NoSQL"
- Cloud: "AWS" ↔ "Amazon Web Services", "Azure" ↔ "Microsoft Cloud"
- Methodologies: "Agile" ↔ "Scrum, Kanban", "DevOps" ↔ "CI/CD"
""",
            "Marketing": """
NGÀNH MARKETING & DIGITAL:
- Platforms: "Facebook Ads" ↔ "Social Media Marketing", "Google Ads" ↔ "PPC"
- Tools: "Canva" ↔ "Graphic Design", "Capcut" ↔ "Video Editing", "Figma" ↔ "UI/UX Design"
- Skills: "SEO" ↔ "Search Engine Optimization", "Content Creation" ↔ "Copywriting"
- Analytics: "Google Analytics" ↔ "Data Analysis", "Facebook Insights" ↔ "Social Analytics"
- Campaigns: "Email Marketing" ↔ "Email Campaigns", "Influencer Marketing" ↔ "KOL"
""",
            "Finance": """
NGÀNH TÀI CHÍNH & KẾ TOÁN:
- Software: "Excel" ↔ "Spreadsheets", "QuickBooks" ↔ "Accounting Software", "SAP" ↔ "ERP"
- Skills: "Financial Modeling" ↔ "Excel Advanced", "Budgeting" ↔ "Financial Planning"
- Analysis: "Data Analysis" ↔ "Financial Analysis", "Audit" ↔ "Compliance"
- Tools: "Power BI" ↔ "Business Intelligence", "Tableau" ↔ "Data Visualization"
""",
            "HR": """
NGÀNH NHÂN SỰ & TUYỂN DỤNG:
- Processes: "Recruitment" ↔ "Talent Acquisition", "Onboarding" ↔ "Employee Orientation"
- Systems: "HRIS" ↔ "HR Software", "ATS" ↔ "Applicant Tracking"
- Skills: "Employee Relations" ↔ "HR Management", "Performance Review" ↔ "HR Operations"
- Training: "Training" ↔ "Learning & Development", "Compensation" ↔ "Payroll"
""",
            "Sales": """
NGÀNH BÁN HÀNG & KINH DOANH:
- CRM: "Salesforce" ↔ "CRM", "HubSpot" ↔ "Lead Management"
- Skills: "Negotiation" ↔ "Sales Skills", "Customer Service" ↔ "Client Relations"
- Process: "Lead Generation" ↔ "Prospecting", "Sales Pipeline" ↔ "Sales Process"
- Tools: "LinkedIn" ↔ "B2B Sales", "Cold Calling" ↔ "Outbound Sales"
""",
            "Design": """
NGÀNH THIẾT KẾ & SÁNG TẠO:
- Tools: "Photoshop" ↔ "Adobe Creative Suite", "Figma" ↔ "UI/UX Design"
- Skills: "Graphic Design" ↔ "Visual Design", "Brand Design" ↔ "Brand Identity"
- Software: "Illustrator" ↔ "Vector Graphics", "InDesign" ↔ "Layout Design"
- Digital: "Digital Art" ↔ "Illustration", "Web Design" ↔ "UI Design"
""",
            "Healthcare": """
NGÀNH Y TẾ & CHĂM SÓC SỨC KHỎE:
- Systems: "EMR" ↔ "Electronic Medical Records", "EHR" ↔ "Health Records"
- Skills: "Patient Care" ↔ "Healthcare", "Clinical Documentation" ↔ "Medical Records"
- Tools: "Epic" ↔ "Healthcare Software", "Cerner" ↔ "Hospital Systems"
- Processes: "Patient Assessment" ↔ "Medical Evaluation", "Treatment Planning" ↔ "Care Plans"
""",
            "Education": """
NGÀNH GIÁO DỤC & ĐÀO TẠO:
- Platforms: "LMS" ↔ "Learning Management System", "Moodle" ↔ "E-learning"
- Skills: "Curriculum Development" ↔ "Educational Design", "Student Assessment" ↔ "Academic Evaluation"
- Tools: "Google Classroom" ↔ "Online Teaching", "Zoom" ↔ "Virtual Learning"
- Methods: "Blended Learning" ↔ "Hybrid Education", "Student Engagement" ↔ "Active Learning"
"""
        }
        
        return industry_prompts.get(detected_industry, """
NGÀNH TỔNG QUÁT:
- Tập trung vào semantic similarity và functional equivalents
- Xem xét skill level và industry context
- Chấp nhận partial matches khi có ý nghĩa tương đương
""")
    
    def _find_missing_skills(self, jd_skills: List[str], matched_skills: List[str]) -> List[str]:
        """Tìm skills còn thiếu"""
        matched_lower = [self._normalize_skill(skill) for skill in matched_skills]
        missing = [skill for skill in jd_skills if self._normalize_skill(skill) not in matched_lower]
        return missing
    
    def _calculate_match_score(self, matched_count: int, total_jd_skills: int) -> float:
        """Tính match score"""
        if total_jd_skills == 0:
            return 0.0
        return min(100.0, (matched_count / total_jd_skills) * 100) 