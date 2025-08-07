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
        """Prepare context for LLM với tối ưu cho đa ngành nghề"""
        
        # Extract key information
        cv_skills = cv_analysis.get('skills', [])
        jd_skills = jd_analysis.get('extracted_skills', [])
        matching_skills = matching_analysis.get('matching_skills', [])
        missing_skills = matching_analysis.get('missing_skills', [])
        skills_match_score = matching_analysis.get('match_score', 0)
        
        quality_score = quality_analysis.get('quality_score', 0)
        strengths = quality_analysis.get('strengths', [])
        weaknesses = quality_analysis.get('weaknesses', [])
        
        # Thêm thông tin chi tiết về CV
        cv_job_title = cv_analysis.get('job_title', 'N/A')
        cv_experience = cv_analysis.get('experience', [])
        cv_education = cv_analysis.get('education', [])
        cv_projects = cv_analysis.get('projects', [])
        
        # Industry-specific guidance
        industry_guidance = self._get_industry_guidance(job_category, job_position)
        
        context = f"""
PHÂN TÍCH CV-JD CHO VỊ TRÍ: {job_position} - NGÀNH: {job_category}

THÔNG TIN CV:
- Job Title hiện tại: {cv_job_title}
- Kỹ năng CV: {', '.join(cv_skills[:15])}
- Kinh nghiệm: {len(cv_experience)} vị trí
- Dự án: {len(cv_projects)} dự án
- Điểm chất lượng CV: {quality_score:.2f}

THÔNG TIN JD:
- Kỹ năng yêu cầu: {', '.join(jd_skills[:15])}

KẾT QUẢ SO KHỚP:
- Kỹ năng phù hợp: {', '.join(matching_skills[:10])}
- Kỹ năng thiếu: {', '.join(missing_skills[:10])}
- Tỷ lệ khớp: {skills_match_score:.1f}%
- Điểm tổng thể: {overall_score:.1f}/100

YÊU CẦU FEEDBACK:
1. Viết 2-3 câu tổng hợp điểm mạnh về kỹ năng, nêu rõ 3-4 kỹ năng phù hợp nổi bật nhất.
2. Viết 2-3 câu tổng hợp điểm cần cải thiện, nêu rõ 3-4 kỹ năng thiếu quan trọng nhất.
3. Gợi ý cải thiện cụ thể (3-4 gợi ý ngắn gọn).
4. Đảm bảo feedback đa chiều nhưng ngắn gọn, dễ hiểu.
5. Trả về JSON format đầy đủ:
{{
  "overall_assessment": "Đánh giá tổng quan ngắn gọn (1-2 câu)",
  "strengths": ["2-3 câu tổng hợp điểm mạnh, nêu rõ 3-4 kỹ năng phù hợp nổi bật"],
  "weaknesses": ["2-3 câu tổng hợp điểm cần cải thiện, nêu rõ 3-4 kỹ năng thiếu quan trọng"],
  "specific_suggestions": ["3-4 gợi ý cải thiện ngắn gọn và cụ thể"],
  "priority_actions": ["2-3 hành động ưu tiên với timeline ngắn"],
  "encouragement": "Lời động viên ngắn gọn và thực tế"
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
                        Hãy luôn trả về JSON format chính xác và đầy đủ."""
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.7,
                max_tokens=2000  # Tăng từ 1000 lên 2000
            )
            
            content = response.choices[0].message.content
            logger.info(f"LLM Response: {content}")
            
            # Parse JSON response với error handling tốt hơn
            import json
            try:
                # Kiểm tra xem response có bị cắt không
                if not content.strip().endswith('}'):
                    logger.warning("LLM response appears to be truncated, using fallback")
                    return self._generate_fallback_feedback({}, {}, {}, {}, 0)
                
                feedback_data = json.loads(content)
                
                # Validate required fields
                required_fields = ['overall_assessment', 'strengths', 'weaknesses', 'specific_suggestions', 'priority_actions', 'encouragement']
                for field in required_fields:
                    if field not in feedback_data:
                        logger.warning(f"Missing required field '{field}' in LLM response, using fallback")
                        return self._generate_fallback_feedback({}, {}, {}, {}, 0)
                
                return feedback_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Response content: {content}")
                return self._generate_fallback_feedback({}, {}, {}, {}, 0)
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return self._generate_fallback_feedback({}, {}, {}, {}, 0)
    
    def _generate_fallback_feedback(self, cv_analysis: Dict, jd_analysis: Dict, matching_analysis: Dict, quality_analysis: Dict, overall_score: float) -> Dict:
        """Generate fallback feedback when LLM is not available"""
        
        matching_skills = matching_analysis.get('matching_skills', [])
        missing_skills = matching_analysis.get('missing_skills', [])
        
        if overall_score >= 80:
            assessment = "CV của bạn rất phù hợp với vị trí này! Bạn đã có nền tảng vững chắc và các kỹ năng cần thiết. Với sự chuẩn bị tốt như vậy, bạn có đủ điều kiện để ứng tuyển thành công."
            encouragement = "Hãy tự tin vào khả năng của mình và tiếp tục phát triển để đạt được những thành công lớn hơn trong sự nghiệp."
        elif overall_score >= 60:
            assessment = "CV của bạn khá tốt với nhiều điểm mạnh đáng kể. Tuy nhiên, vẫn còn một số điểm cần cải thiện để tăng cơ hội thành công. Với sự nỗ lực đúng hướng, bạn sẽ có cơ hội tốt."
            encouragement = "Đừng nản lòng, hãy xem những điểm cần cải thiện như cơ hội để phát triển và hoàn thiện bản thân."
        elif overall_score >= 40:
            assessment = "CV của bạn có một số điểm mạnh nhưng cần được cải thiện đáng kể để tăng cơ hội. Việc tập trung vào những kỹ năng thiếu sẽ giúp bạn tiến bộ nhanh chóng."
            encouragement = "Mỗi bước cải thiện đều đáng giá, hãy kiên trì và bạn sẽ thấy kết quả tích cực."
        else:
            assessment = "CV của bạn cần được cải thiện đáng kể để phù hợp với yêu cầu công việc. Tuy nhiên, đây là cơ hội để bạn xây dựng lại CV một cách có hệ thống và chuyên nghiệp hơn."
            encouragement = "Hãy xem đây là cơ hội để học hỏi và phát triển. Mọi chuyên gia đều bắt đầu từ những bước đầu tiên."
        
        # Tạo strengths đa chiều
        strengths = []
        if matching_skills:
            top_matching = matching_skills[:6]  # Lấy 6 kỹ năng phù hợp nhất
            strengths.append(f"Bạn có các kỹ năng phù hợp tốt: {', '.join(top_matching[:3])}. Những kỹ năng này cho thấy bạn đã có nền tảng vững chắc cho vị trí này.")
            if len(top_matching) > 3:
                strengths.append(f"Ngoài ra, bạn còn có kinh nghiệm với {', '.join(top_matching[3:])}, điều này giúp bạn nổi bật so với các ứng viên khác.")
            strengths.append("Việc có nhiều kỹ năng đa dạng như vậy cho thấy khả năng thích ứng và học hỏi nhanh của bạn.")
        
        # Tạo weaknesses đa chiều
        weaknesses = []
        if missing_skills:
            top_missing = missing_skills[:6]  # Lấy 6 kỹ năng thiếu quan trọng nhất
            weaknesses.append(f"Bạn cần bổ sung các kỹ năng quan trọng: {', '.join(top_missing[:3])}. Việc thiếu những kỹ năng này có thể làm giảm cơ hội thành công.")
            if len(top_missing) > 3:
                weaknesses.append(f"Đặc biệt, {', '.join(top_missing[3:])} là những kỹ năng rất được ưa chuộng trong ngành hiện nay.")
            weaknesses.append("Việc phát triển những kỹ năng này sẽ tăng đáng kể giá trị của bạn trong mắt nhà tuyển dụng.")
        
        # Tạo suggestions chi tiết và đa chiều
        suggestions = []
        if missing_skills:
            suggestions.append("Hãy làm nổi bật các kỹ năng phù hợp trong CV và thêm các dự án thực tế để chứng minh khả năng")
            suggestions.append("Bổ sung thêm kinh nghiệm thực tế thông qua các dự án freelance hoặc open-source")
            suggestions.append("Cập nhật CV theo từng vị trí ứng tuyển cụ thể, tập trung vào keywords phù hợp")
            suggestions.append("Tham gia các khóa học online và lấy chứng chỉ để củng cố kiến thức")
            suggestions.append("Xây dựng portfolio với các dự án demo để showcase kỹ năng")
        
        # Tạo priority actions với timeline
        priority_actions = []
        if missing_skills:
            priority_actions.append("Ưu tiên học và thực hành các kỹ năng thiếu trong 2-3 tháng tới")
            priority_actions.append("Tạo dự án portfolio trong 1 tháng để chứng minh kỹ năng")
            priority_actions.append("Tham gia các cộng đồng developer để networking và học hỏi kinh nghiệm")
        
        return {
            "overall_assessment": assessment,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "specific_suggestions": suggestions,
            "priority_actions": priority_actions,
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

    def _get_industry_guidance(self, job_category: str, job_position: str) -> str:
        """Tạo industry-specific guidance cho từng ngành nghề"""
        
        guidance_map = {
            "INFORMATION-TECHNOLOGY": f"""
NGÀNH CÔNG NGHỆ THÔNG TIN:
- Ưu tiên technical skills và latest technologies
- Chú ý framework versions và tool ecosystems
- Soft skills: Problem-solving, analytical thinking
- Trends: Cloud, AI/ML, DevOps, Security
- Learning paths: Online courses, certifications, hands-on projects
- Portfolio: GitHub, technical blogs, open-source contributions
- Salary ranges: Junior (15-25M), Mid (25-40M), Senior (40-70M+)
- Career progression: Junior → Mid → Senior → Lead → Architect
""",
            "MARKETING": f"""
NGÀNH MARKETING & DIGITAL:
- Ưu tiên digital marketing skills và data analytics
- Chú ý platform-specific knowledge (Google, Facebook, TikTok)
- Soft skills: Creativity, communication, data-driven thinking
- Trends: Social commerce, influencer marketing, automation
- Learning paths: Certifications (Google Ads, Facebook Blueprint), practical campaigns
- Portfolio: Campaign case studies, ROI reports, creative samples
- Salary ranges: Junior (12-20M), Mid (20-35M), Senior (35-50M+)
- Career progression: Marketing Assistant → Specialist → Manager → Director
""",
            "FINANCE": f"""
NGÀNH TÀI CHÍNH & KẾ TOÁN:
- Ưu tiên financial modeling và regulatory knowledge
- Chú ý software proficiency (Excel, SAP, Oracle)
- Soft skills: Attention to detail, analytical thinking, compliance
- Trends: Fintech, automation, ESG investing
- Learning paths: Professional certifications (CFA, CPA), industry experience
- Portfolio: Financial models, analysis reports, compliance documentation
- Salary ranges: Junior (15-25M), Mid (25-40M), Senior (40-60M+)
- Career progression: Analyst → Senior Analyst → Manager → Director
""",
            "HUMAN-RESOURCES": f"""
NGÀNH NHÂN SỰ & TUYỂN DỤNG:
- Ưu tiên HR processes và employee relations
- Chú ý HRIS systems và compliance knowledge
- Soft skills: Empathy, communication, conflict resolution
- Trends: Remote work, employee experience, AI in HR
- Learning paths: HR certifications, psychology courses, legal knowledge
- Portfolio: HR policies, employee programs, recruitment strategies
- Salary ranges: Junior (12-20M), Mid (20-35M), Senior (35-50M+)
- Career progression: HR Assistant → Specialist → Manager → Director
""",
            "SALES": f"""
NGÀNH BÁN HÀNG & KINH DOANH:
- Ưu tiên sales techniques và customer relationship
- Chú ý CRM systems và sales methodologies
- Soft skills: Negotiation, persuasion, relationship building
- Trends: Digital sales, consultative selling, customer success
- Learning paths: Sales training, industry knowledge, networking
- Portfolio: Sales achievements, client testimonials, pipeline management
- Salary ranges: Junior (15-25M), Mid (25-45M), Senior (45-80M+)
- Career progression: Sales Rep → Senior Rep → Manager → Director
""",
            "HEALTHCARE": f"""
NGÀNH Y TẾ & CHĂM SÓC SỨC KHỎE:
- Ưu tiên clinical knowledge và patient care
- Chú ý EMR systems và medical protocols
- Soft skills: Empathy, patience, attention to detail
- Trends: Telemedicine, AI diagnostics, personalized medicine
- Learning paths: Medical certifications, continuing education, specialization
- Portfolio: Clinical experience, patient outcomes, research contributions
- Salary ranges: Junior (20-30M), Mid (30-50M), Senior (50-80M+)
- Career progression: Junior Staff → Senior Staff → Specialist → Director
""",
            "EDUCATION": f"""
NGÀNH GIÁO DỤC & ĐÀO TẠO:
- Ưu tiên pedagogical skills và curriculum development
- Chú ý LMS platforms và assessment methods
- Soft skills: Patience, communication, adaptability
- Trends: EdTech, blended learning, personalized education
- Learning paths: Teaching certifications, subject expertise, technology skills
- Portfolio: Lesson plans, student outcomes, innovative teaching methods
- Salary ranges: Junior (15-25M), Mid (25-40M), Senior (40-60M+)
- Career progression: Teacher → Senior Teacher → Coordinator → Principal
""",
            "DESIGN": f"""
NGÀNH THIẾT KẾ & SÁNG TẠO:
- Ưu tiên design skills và creative thinking
- Chú ý design tools và industry trends
- Soft skills: Creativity, collaboration, client communication
- Trends: UX/UI design, sustainable design, digital transformation
- Learning paths: Design courses, software mastery, portfolio building
- Portfolio: Design projects, client work, creative concepts
- Salary ranges: Junior (15-25M), Mid (25-45M), Senior (45-70M+)
- Career progression: Junior Designer → Senior Designer → Art Director → Creative Director
"""
        }
        
        return guidance_map.get(job_category.upper(), f"""
NGÀNH {job_category.upper()}:
- Tập trung vào industry-specific skills và requirements
- Chú ý đến latest trends và best practices trong ngành
- Soft skills phù hợp với {job_position}
- Learning paths: Industry certifications, practical experience
- Portfolio: Relevant projects và achievements
- Salary ranges: Junior (12-25M), Mid (25-45M), Senior (45-70M+)
- Career progression: Junior → Mid → Senior → Lead → Director
""") 