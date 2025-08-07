import logging
from typing import List, Dict, Optional
import os

logger = logging.getLogger(__name__)

class InterviewSimulator:
    """Service for generating interview questions based on JD and CV analysis"""
    
    def __init__(self):
        # Try to initialize OpenAI client, but provide fallback for testing
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.use_openai = True
            else:
                self.client = None
                self.use_openai = False
                logger.warning("OPENAI_API_KEY not set, using template-based generation")
        except ImportError:
            self.client = None
            self.use_openai = False
            logger.warning("OpenAI not available, using template-based generation")
        
        self.question_templates = {
            "technical": {
                "IT": [
                    "Explain your experience with {skill}",
                    "How would you approach debugging a {skill} issue?",
                    "Describe a project where you used {skill}",
                    "What are the best practices for {skill}?",
                    "How do you stay updated with {skill} trends?"
                ],
                "Marketing": [
                    "How would you create a campaign for {skill}?",
                    "Describe your experience with {skill} tools",
                    "What metrics would you track for {skill}?",
                    "How do you measure ROI for {skill}?",
                    "What trends are you seeing in {skill}?"
                ],
                "Finance": [
                    "How would you analyze {skill} data?",
                    "Describe your experience with {skill} systems",
                    "What regulations apply to {skill}?",
                    "How do you ensure compliance with {skill}?",
                    "What risks are associated with {skill}?"
                ]
            },
            "behavioral": [
                "Describe a challenging project you worked on",
                "Tell me about a time you had to learn something quickly",
                "How do you handle conflicting priorities?",
                "Describe a situation where you had to persuade someone",
                "Tell me about a time you failed and what you learned"
            ],
            "situational": [
                "What would you do if a project deadline was moved up?",
                "How would you handle a difficult team member?",
                "What if you discovered a critical bug in production?",
                "How would you prioritize multiple urgent requests?",
                "What would you do if a client was unhappy with your work?"
            ],
            "culture_fit": [
                "What motivates you in your work?",
                "How do you prefer to receive feedback?",
                "Describe your ideal work environment",
                "How do you handle stress and pressure?",
                "What are your career goals for the next 5 years?"
            ]
        }
    
    def generate_interview_session(
        self,
        job_category: str,
        job_position: str,
        jd_skills: List[str],
        cv_skills: List[str],
        missing_skills: List[str],
        overall_score: float
    ) -> Dict:
        """Generate a complete interview session"""
        
        try:
            # Determine difficulty based on overall score
            difficulty = self._determine_difficulty(overall_score)
            
            # Generate questions
            questions = []
            
            # Technical questions (focus on missing skills and JD skills)
            technical_questions = self._generate_technical_questions(
                job_category, jd_skills, missing_skills, difficulty
            )
            questions.extend(technical_questions)
            
            # Behavioral questions
            behavioral_questions = self._generate_behavioral_questions(difficulty)
            questions.extend(behavioral_questions)
            
            # Situational questions
            situational_questions = self._generate_situational_questions(difficulty)
            questions.extend(situational_questions)
            
            # Culture fit questions
            culture_questions = self._generate_culture_questions(difficulty)
            questions.extend(culture_questions)
            
            # Calculate session details
            total_duration = sum(q.get('time_limit', 120) for q in questions)
            
            session = {
                "session_id": f"session_{int(time.time())}",
                "job_category": job_category,
                "job_position": job_position,
                "difficulty": difficulty,
                "questions": questions,
                "total_duration": total_duration,
                "scoring_criteria": self._get_scoring_criteria(),
                "tips": self._get_interview_tips(difficulty),
                "estimated_questions": len(questions)
            }
            
            logger.info(f"Generated interview session for {job_position} with {len(questions)} questions")
            return session
            
        except Exception as e:
            logger.error(f"Error generating interview session: {e}")
            raise
    
    def _determine_difficulty(self, overall_score: float) -> str:
        """Determine interview difficulty based on CV-JD match score"""
        if overall_score >= 85:
            return "hard"
        elif overall_score >= 70:
            return "medium"
        else:
            return "easy"
    
    def _generate_technical_questions(
        self,
        job_category: str,
        jd_skills: List[str],
        missing_skills: List[str],
        difficulty: str
    ) -> List[Dict]:
        """Generate technical questions based on skills"""
        questions = []
        
        # Focus on missing skills first
        priority_skills = missing_skills + jd_skills[:3]  # Top 3 JD skills
        
        templates = self.question_templates["technical"].get(job_category, 
                   self.question_templates["technical"]["IT"])  # Default to IT
        
        for i, skill in enumerate(priority_skills[:3]):  # Max 3 technical questions
            template = templates[i % len(templates)]
            question_text = template.format(skill=skill)
            
            questions.append({
                "id": len(questions) + 1,
                "type": "technical",
                "question": question_text,
                "skill": skill,
                "difficulty": difficulty,
                "expected_points": self._get_expected_points("technical", skill),
                "time_limit": 180 if difficulty == "hard" else 120,
                "scoring_weight": 0.4
            })
        
        return questions
    
    def _generate_behavioral_questions(self, difficulty: str) -> List[Dict]:
        """Generate behavioral questions"""
        questions = []
        templates = self.question_templates["behavioral"]
        
        for i in range(2):  # 2 behavioral questions
            questions.append({
                "id": len(questions) + 1,
                "type": "behavioral",
                "question": templates[i],
                "difficulty": difficulty,
                "expected_points": self._get_expected_points("behavioral"),
                "time_limit": 180,
                "scoring_weight": 0.25
            })
        
        return questions
    
    def _generate_situational_questions(self, difficulty: str) -> List[Dict]:
        """Generate situational questions"""
        questions = []
        templates = self.question_templates["situational"]
        
        for i in range(2):  # 2 situational questions
            questions.append({
                "id": len(questions) + 1,
                "type": "situational",
                "question": templates[i],
                "difficulty": difficulty,
                "expected_points": self._get_expected_points("situational"),
                "time_limit": 150,
                "scoring_weight": 0.2
            })
        
        return questions
    
    def _generate_culture_questions(self, difficulty: str) -> List[Dict]:
        """Generate culture fit questions"""
        questions = []
        templates = self.question_templates["culture_fit"]
        
        for i in range(1):  # 1 culture fit question
            questions.append({
                "id": len(questions) + 1,
                "type": "culture_fit",
                "question": templates[i],
                "difficulty": difficulty,
                "expected_points": self._get_expected_points("culture_fit"),
                "time_limit": 120,
                "scoring_weight": 0.15
            })
        
        return questions
    
    def _get_expected_points(self, question_type: str, skill: str = None) -> List[str]:
        """Get expected answer points for different question types"""
        points_map = {
            "technical": [
                "Technical knowledge and understanding",
                "Practical experience and examples",
                "Problem-solving approach",
                "Best practices and methodologies"
            ],
            "behavioral": [
                "Specific situation description",
                "Actions taken and reasoning",
                "Results and outcomes",
                "Lessons learned"
            ],
            "situational": [
                "Analysis of the situation",
                "Decision-making process",
                "Action plan and execution",
                "Risk assessment and mitigation"
            ],
            "culture_fit": [
                "Personal values and motivations",
                "Work style and preferences",
                "Team collaboration approach",
                "Career goals and aspirations"
            ]
        }
        
        return points_map.get(question_type, ["Relevance", "Clarity", "Depth"])
    
    def _get_scoring_criteria(self) -> Dict:
        """Get scoring criteria for different aspects"""
        return {
            "technical_accuracy": 0.4,
            "communication": 0.25,
            "problem_solving": 0.2,
            "culture_fit": 0.15
        }
    
    def _get_interview_tips(self, difficulty: str) -> List[str]:
        """Get interview tips based on difficulty"""
        tips = [
            "Be specific with examples from your experience",
            "Use the STAR method for behavioral questions",
            "Show enthusiasm and confidence",
            "Ask thoughtful questions about the role"
        ]
        
        if difficulty == "hard":
            tips.extend([
                "Demonstrate deep technical knowledge",
                "Provide detailed technical explanations",
                "Show advanced problem-solving skills"
            ])
        elif difficulty == "easy":
            tips.extend([
                "Focus on learning potential",
                "Show willingness to learn new skills",
                "Emphasize transferable skills"
            ])
        
        return tips

# Import time for session ID generation
import time 