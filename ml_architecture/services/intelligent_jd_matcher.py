import openai
import os
import logging
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from dotenv import load_dotenv

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
        """Tìm semantic matches sử dụng LLM"""
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
            Phân tích semantic similarity giữa skills trong CV và JD.
            
            Skills trong CV: {', '.join(cv_remaining)}
            Skills trong JD: {', '.join(jd_remaining)}
            
            Tìm các skill từ CV có thể match với JD dựa trên semantic similarity.
            
            Ví dụ matching:
            - ".NET Stack" match "C# .NET Core 6"
            - "React" match "Frontend Development"
            - "MongoDB" match "Database Management"
            - "Git" match "Version Control"
            - "Photoshop" match "Graphic Design"
            - "Capcut" match "Video Editing"
            - "Excel" match "Data Analysis"
            - "Facebook Ads" match "Digital Marketing"
            - "SEO" match "Search Engine Optimization"
            - "Recruitment" match "Talent Acquisition"
            
            Trả về tên skill từ CV có thể match, mỗi skill một dòng:
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse skills from LLM response (format: "- "Skill" match "JD"" or "- Skill match JD")
            matched_skills = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('- ') and 'match' in line:
                    # Extract skill name from format: "- "Skill" match "JD"" or "- Skill match JD"
                    try:
                        # Try with quotes first
                        if '"' in line:
                            skill_start = line.find('"') + 1
                            skill_end = line.find('"', skill_start)
                            if skill_start > 0 and skill_end > skill_start:
                                skill_name = line[skill_start:skill_end]
                                matched_skills.append(skill_name)
                        else:
                            # Try without quotes: "- Skill match JD"
                            parts = line.split('match')
                            if len(parts) >= 2:
                                skill_part = parts[0].replace('- ', '').strip()
                                matched_skills.append(skill_part)
                    except:
                        continue
            
            # Filter to only include skills that are actually in CV
            semantic_matches = [skill for skill in matched_skills if skill in cv_remaining]
            
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