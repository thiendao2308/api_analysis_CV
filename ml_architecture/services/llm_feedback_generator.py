import openai
import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
logger = logging.getLogger(__name__)

class LLMFeedbackGenerator:
    """LLM-powered feedback generator for intelligent CV suggestions"""
    
    def __init__(self):
        self.client = None
        if OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("✅ LLM Feedback Generator initialized with OpenAI")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
    
    def generate_intelligent_feedback(self, 
                                   cv_analysis: Dict,
                                   jd_analysis: Dict,
                                   matching_analysis: Dict,
                                   quality_analysis: Dict,
                                   overall_score: float,
                                   job_category: str,
                                   job_position: str) -> Dict:
        """
        Generate intelligent feedback using LLM
        """
        if not self.client:
            logger.warning("OpenAI client not available, using fallback feedback")
            return self._generate_fallback_feedback(cv_analysis, jd_analysis, matching_analysis, quality_analysis, overall_score)
        
        try:
            # Prepare context for LLM
            context = self._prepare_context(cv_analysis, jd_analysis, matching_analysis, quality_analysis, overall_score, job_category, job_position)
            
            # Generate feedback using LLM
            feedback_result = self._call_llm_for_feedback(context)
            
            return feedback_result
            
        except Exception as e:
            logger.error(f"❌ Error generating LLM feedback: {e}")
            return self._generate_fallback_feedback(cv_analysis, jd_analysis, matching_analysis, quality_analysis, overall_score)
    
    def _prepare_context(self, cv_analysis: Dict, jd_analysis: Dict, matching_analysis: Dict, quality_analysis: Dict, overall_score: float, job_category: str, job_position: str) -> str:
        """Prepare context for LLM"""
        
        # Extract key information
        cv_skills = cv_analysis.get('skills', [])
        jd_skills = jd_analysis.get('extracted_skills', [])
        matching_skills = matching_analysis.get('matching_skills', [])
        missing_skills = matching_analysis.get('missing_skills', [])
        skills_match_score = matching_analysis.get('skills_match_score', 0)
        
        quality_score = quality_analysis.get('quality_score', 0)
        strengths = quality_analysis.get('strengths', [])
        weaknesses = quality_analysis.get('weaknesses', [])
        
        # Thêm thông tin chi tiết về CV
        cv_job_title = cv_analysis.get('job_title', 'N/A')
        cv_experience = cv_analysis.get('experience', [])
        cv_education = cv_analysis.get('education', [])
        cv_projects = cv_analysis.get('projects', [])
        
        context = f"""
PHÂN TÍCH CV-JD CHO VỊ TRÍ: {job_position} - NGÀNH: {job_category}

THÔNG TIN CV:
- Job Title hiện tại: {cv_job_title}
- Kỹ năng CV: {', '.join(cv_skills[:15])}
- Kinh nghiệm: {len(cv_experience)} vị trí
- Dự án: {len(cv_projects)} dự án
- Điểm chất lượng CV: {quality_score:.2f}
- Điểm mạnh: {', '.join(strengths[:5])}
- Điểm yếu: {', '.join(weaknesses[:5])}

THÔNG TIN JD:
- Kỹ năng yêu cầu: {', '.join(jd_skills[:15])}

KẾT QUẢ SO KHỚP:
- Kỹ năng phù hợp: {', '.join(matching_skills[:10])}
- Kỹ năng thiếu: {', '.join(missing_skills[:10])}
- Tỷ lệ khớp: {skills_match_score:.1f}%
- Điểm tổng thể: {overall_score:.1f}/100

PHÂN TÍCH CHI TIẾT:
- CV có {len(cv_skills)} kỹ năng, JD yêu cầu {len(jd_skills)} kỹ năng
- Khớp chính xác: {len(matching_skills)}/{len(jd_skills)} kỹ năng
- Thiếu {len(missing_skills)} kỹ năng quan trọng

YÊU CẦU: Hãy đưa ra feedback chân thật, cụ thể và hữu ích cho ứng viên. Feedback phải:
1. Chân thật - không quá lạc quan hay bi quan
2. Cụ thể - chỉ ra điểm mạnh/yếu cụ thể dựa trên missing skills
3. Hữu ích - đưa ra gợi ý thực tế để cải thiện missing skills
4. Cân bằng - vừa động viên vừa chỉ ra điểm cần cải thiện
5. Phù hợp với ngành nghề {job_category} và vị trí {job_position}

Đặc biệt chú ý:
- Nếu missing skills nhiều: đưa ra lộ trình học tập cụ thể
- Nếu matching skills ít: gợi ý cách highlight skills hiện có
- Nếu overall score thấp: đưa ra priority actions rõ ràng

Hãy trả về JSON format:
{{
    "overall_assessment": "Đánh giá tổng quan chân thật dựa trên missing skills",
    "strengths": ["Điểm mạnh cụ thể 1", "Điểm mạnh cụ thể 2"],
    "weaknesses": ["Điểm yếu cụ thể dựa trên missing skills 1", "Điểm yếu cụ thể 2"],
    "specific_suggestions": ["Gợi ý cụ thể để học missing skills 1", "Gợi ý cụ thể 2"],
    "priority_actions": ["Hành động ưu tiên để cải thiện 1", "Hành động ưu tiên 2"],
    "encouragement": "Lời động viên chân thành dựa trên potential"
}}
"""
        return context
    
    def _call_llm_for_feedback(self, context: str) -> Dict:
        """Call LLM to generate feedback"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là một chuyên gia tư vấn CV với 10+ năm kinh nghiệm. 
                        Bạn có khả năng đánh giá chân thật và đưa ra gợi ý hữu ích cho ứng viên.
                        Hãy luôn trả về JSON format chính xác."""
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            logger.info(f"LLM Response: {content}")
            
            # Parse JSON response
            import json
            try:
                feedback_data = json.loads(content)
                return feedback_data
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return self._generate_fallback_feedback({}, {}, {}, {}, 0)
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return self._generate_fallback_feedback({}, {}, {}, {}, 0)
    
    def _generate_fallback_feedback(self, cv_analysis: Dict, jd_analysis: Dict, matching_analysis: Dict, quality_analysis: Dict, overall_score: float) -> Dict:
        """Generate fallback feedback when LLM is not available"""
        
        matching_skills = matching_analysis.get('matching_skills', [])
        missing_skills = matching_analysis.get('missing_skills', [])
        
        if overall_score >= 80:
            assessment = "CV của bạn rất phù hợp với vị trí này!"
            encouragement = "Bạn có đủ điều kiện để ứng tuyển thành công."
        elif overall_score >= 60:
            assessment = "CV của bạn khá tốt, chỉ cần cải thiện một số điểm."
            encouragement = "Với một chút cải thiện, bạn sẽ có cơ hội tốt."
        elif overall_score >= 40:
            assessment = "CV của bạn cần được cải thiện để tăng cơ hội."
            encouragement = "Đừng nản lòng, hãy tập trung vào việc cải thiện."
        else:
            assessment = "CV của bạn cần được cải thiện đáng kể."
            encouragement = "Hãy xem đây là cơ hội để học hỏi và phát triển."
        
        strengths = []
        if matching_skills:
            strengths.append(f"Bạn đã có các kỹ năng phù hợp: {', '.join(matching_skills[:3])}")
        
        weaknesses = []
        if missing_skills:
            weaknesses.append(f"Cần bổ sung: {', '.join(missing_skills[:3])}")
        
        suggestions = [
            "Hãy làm nổi bật các kỹ năng phù hợp trong CV",
            "Bổ sung thêm kinh nghiệm thực tế nếu có thể",
            "Cập nhật CV theo từng vị trí ứng tuyển cụ thể"
        ]
        
        return {
            "overall_assessment": assessment,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "specific_suggestions": suggestions,
            "priority_actions": ["Cải thiện CV", "Bổ sung kỹ năng"],
            "encouragement": encouragement
        }
    
    def generate_quick_feedback(self, overall_score: float, job_category: str) -> str:
        """Generate quick feedback for simple cases"""
        if not self.client:
            return self._generate_quick_fallback(overall_score)
        
        try:
            prompt = f"""
Điểm CV: {overall_score}/100
Ngành nghề: {job_category}

Hãy đưa ra 1-2 câu feedback ngắn gọn, chân thật và động viên cho ứng viên.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia tư vấn CV. Hãy đưa ra feedback ngắn gọn, chân thật và động viên."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating quick feedback: {e}")
            return self._generate_quick_fallback(overall_score)
    
    def _generate_quick_fallback(self, overall_score: float) -> str:
        """Generate quick fallback feedback"""
        if overall_score >= 80:
            return "CV của bạn rất tốt! Hãy tự tin ứng tuyển."
        elif overall_score >= 60:
            return "CV khá tốt, chỉ cần cải thiện một số điểm nhỏ."
        elif overall_score >= 40:
            return "CV cần được cải thiện để tăng cơ hội thành công."
        else:
            return "CV cần được cải thiện đáng kể. Hãy tập trung vào việc phát triển kỹ năng." 