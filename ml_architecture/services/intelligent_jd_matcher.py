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
    """Intelligent JD matching sử dụng LLM để hiểu semantic similarity"""
    
    def __init__(self):
        self.client = None
        if OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("✅ Intelligent JD Matcher initialized with OpenAI")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
    
    def intelligent_matching(self, cv_skills: List[str], jd_skills: List[str]) -> Dict:
        """Intelligent matching giữa CV skills và JD skills"""
        logger.info(f"Starting intelligent JD matching: CV={len(cv_skills)} skills, JD={len(jd_skills)} skills")
        
        # 1. Exact matches
        exact_matches = self._find_exact_matches(cv_skills, jd_skills)
        
        # 2. Semantic matches (LLM-based)
        semantic_matches = self._find_semantic_matches(cv_skills, jd_skills, exact_matches)
        
        # 3. Calculate scores
        all_matches = exact_matches + semantic_matches
        missing_skills = self._find_missing_skills(jd_skills, all_matches)
        
        # 4. Calculate match score
        match_score = self._calculate_match_score(len(all_matches), len(jd_skills))
        
        return {
            "matching_skills": all_matches,
            "missing_skills": missing_skills,
            "exact_matches": exact_matches,
            "semantic_matches": semantic_matches,
            "match_score": match_score,
            "total_cv_skills": len(cv_skills),
            "total_jd_skills": len(jd_skills)
        }
    
    def _find_exact_matches(self, cv_skills: List[str], jd_skills: List[str]) -> List[str]:
        """Tìm exact matches"""
        cv_skills_lower = [skill.lower() for skill in cv_skills]
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        
        exact_matches = []
        for cv_skill, cv_lower in zip(cv_skills, cv_skills_lower):
            if cv_lower in jd_skills_lower:
                exact_matches.append(cv_skill)
        
        logger.info(f"Found {len(exact_matches)} exact matches")
        return exact_matches
    
    def _find_semantic_matches(self, cv_skills: List[str], jd_skills: List[str], exclude_matches: List[str]) -> List[str]:
        """Tìm semantic matches sử dụng LLM với prompt tối ưu cho đa ngành nghề"""
        if not self.client:
            logger.warning("OpenAI client not available, skipping semantic matching")
            return []
        
        semantic_matches = []
        exclude_lower = [skill.lower() for skill in exclude_matches]
        
        # Group skills for batch processing
        cv_remaining = [skill for skill in cv_skills if skill.lower() not in exclude_lower]
        jd_remaining = [skill for skill in jd_skills if skill.lower() not in exclude_lower]
        
        if not cv_remaining or not jd_remaining:
            return semantic_matches
        
        try:
            prompt = f"""
Bạn là chuyên gia phân tích skills matching cho nhiều ngành nghề. Hãy phân tích semantic similarity giữa skills CV và JD.

SKILLS CV: {', '.join(cv_remaining)}
SKILLS JD: {', '.join(jd_remaining)}

QUY TẮC MATCHING THEO NGÀNH NGHỀ:

1. CÔNG NGHỆ THÔNG TIN:
   - Technology Families: ".NET Stack" ↔ "C# .NET Core 6", "React" ↔ "Vue.js", "MongoDB" ↔ "Database Management"
   - Framework Equivalents: "ASP.NET Core MVC" ↔ ".NET Framework", "Express.js" ↔ "Node.js"
   - Tool Equivalents: "Git" ↔ "Version Control", "Postman" ↔ "API Testing"

2. MARKETING & DIGITAL:
   - Platform Equivalents: "Facebook Ads" ↔ "Social Media Marketing", "Google Ads" ↔ "PPC Campaigns"
   - Tool Equivalents: "Canva" ↔ "Graphic Design", "Capcut" ↔ "Video Editing"
   - Skill Equivalents: "SEO" ↔ "Search Engine Optimization", "Content Creation" ↔ "Copywriting"

3. TÀI CHÍNH & KẾ TOÁN:
   - Software Equivalents: "Excel" ↔ "Data Analysis", "QuickBooks" ↔ "Accounting Software"
   - Skill Equivalents: "Financial Modeling" ↔ "Excel Advanced", "Audit" ↔ "Financial Analysis"

4. NHÂN SỰ & TUYỂN DỤNG:
   - Process Equivalents: "Recruitment" ↔ "Talent Acquisition", "HRIS" ↔ "HR Software"
   - Skill Equivalents: "Employee Relations" ↔ "HR Management", "Performance Review" ↔ "HR Operations"

5. THIẾT KẾ & SÁNG TẠO:
   - Tool Equivalents: "Photoshop" ↔ "Graphic Design", "Figma" ↔ "UI/UX Design"
   - Skill Equivalents: "Brand Design" ↔ "Visual Identity", "Illustration" ↔ "Digital Art"

6. BÁN HÀNG & KINH DOANH:
   - Process Equivalents: "CRM" ↔ "Customer Relationship Management", "Sales Pipeline" ↔ "Lead Management"
   - Skill Equivalents: "Negotiation" ↔ "Sales Skills", "Market Research" ↔ "Business Analysis"

7. Y TẾ & CHĂM SÓC SỨC KHỎE:
   - System Equivalents: "EMR" ↔ "Electronic Medical Records", "Patient Care" ↔ "Healthcare Management"
   - Skill Equivalents: "Clinical Documentation" ↔ "Medical Records", "Patient Assessment" ↔ "Healthcare"

8. GIÁO DỤC & ĐÀO TẠO:
   - Platform Equivalents: "LMS" ↔ "Learning Management System", "Online Teaching" ↔ "E-learning"
   - Skill Equivalents: "Curriculum Development" ↔ "Educational Design", "Student Assessment" ↔ "Academic Evaluation"

NGUYÊN TẮC CHUNG:
- Ưu tiên matching theo technology families và platform equivalents
- Xem xét skill level tương đương (basic ↔ intermediate ↔ advanced)
- Chấp nhận partial matches khi có semantic similarity cao
- Loại trừ matches quá chung chung hoặc không liên quan

TRẢ VỀ: Chỉ tên skill từ CV có thể match, mỗi skill một dòng, format: "Skill Name"
Ví dụ:
"ASP.NET Core MVC"
"Git"
"ReactJS"
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
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
            # Look for patterns like "Skill Name" or "Skill Name match" or "- Skill Name"
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
    
    def _find_missing_skills(self, jd_skills: List[str], matched_skills: List[str]) -> List[str]:
        """Tìm skills còn thiếu"""
        matched_lower = [skill.lower() for skill in matched_skills]
        missing = [skill for skill in jd_skills if skill.lower() not in matched_lower]
        return missing
    
    def _calculate_match_score(self, matched_count: int, total_jd_skills: int) -> float:
        """Tính match score"""
        if total_jd_skills == 0:
            return 0.0
        return (matched_count / total_jd_skills) * 100 