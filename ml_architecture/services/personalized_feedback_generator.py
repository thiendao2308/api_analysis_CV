import logging
from typing import Dict, Optional
from .llm_personal_info_extractor import PersonalInfo

logger = logging.getLogger(__name__)

class PersonalizedFeedbackGenerator:
    """Tạo câu đánh giá chung đơn giản cho ứng viên"""
    
    def __init__(self):
        # Template đơn giản cho câu đánh giá
        self.assessment_templates = {
            "excellent": [
                "Tôi đánh giá CV của bạn rất phù hợp với công việc",
                "CV của bạn hoàn toàn phù hợp với yêu cầu công việc",
                "Tôi rất ấn tượng với CV của bạn, rất phù hợp với vị trí này"
            ],
            "good": [
                "Tôi đánh giá CV của bạn khá phù hợp với công việc",
                "CV của bạn tương đối phù hợp với yêu cầu công việc",
                "Tôi thấy CV của bạn có tiềm năng tốt cho vị trí này"
            ],
            "fair": [
                "Tôi đánh giá CV của bạn cần cải thiện để phù hợp hơn với công việc",
                "CV của bạn cần bổ sung thêm để đáp ứng yêu cầu công việc",
                "Tôi thấy CV của bạn cần phát triển thêm để phù hợp với vị trí này"
            ],
            "poor": [
                "Tôi đánh giá CV của bạn cần cải thiện đáng kể để phù hợp với công việc",
                "CV của bạn cần nhiều cải thiện để đáp ứng yêu cầu công việc",
                "Tôi thấy CV của bạn cần phát triển nhiều hơn để phù hợp với vị trí này"
            ]
        }
        
        self.reason_templates = {
            "excellent": [
                "vì bạn có đầy đủ kỹ năng cần thiết và kinh nghiệm phù hợp",
                "vì profile của bạn rất phù hợp với yêu cầu của công ty",
                "vì bạn có background và expertise phù hợp với vị trí này"
            ],
            "good": [
                "vì bạn có nhiều kỹ năng phù hợp, chỉ cần bổ sung một số điểm",
                "vì background của bạn khá phù hợp, cần phát triển thêm một số kỹ năng",
                "vì bạn có tiềm năng tốt, chỉ cần cải thiện một số khía cạnh"
            ],
            "fair": [
                "vì bạn còn thiếu một số kỹ năng quan trọng cần thiết cho vị trí này",
                "vì profile của bạn cần phát triển thêm để đáp ứng đầy đủ yêu cầu",
                "vì bạn cần bổ sung thêm kinh nghiệm và kỹ năng chuyên môn"
            ],
            "poor": [
                "vì bạn còn thiếu nhiều kỹ năng cơ bản cần thiết cho vị trí này",
                "vì profile của bạn cần phát triển đáng kể để đáp ứng yêu cầu công việc",
                "vì bạn cần học hỏi và phát triển nhiều kỹ năng chuyên môn"
            ]
        }
    
    def generate_personalized_feedback(self, 
                                     personal_info: PersonalInfo,
                                     analysis_result: Dict,
                                     job_position: str,
                                     job_category: str) -> Dict:
        """Tạo câu đánh giá chung đơn giản"""
        try:
            # Xác định level đánh giá dựa trên overall score
            overall_score = analysis_result.get('overall_score', 0)
            assessment_level = self._get_assessment_level(overall_score)
            
            # Tạo câu đánh giá đơn giản
            greeting = f"Xin chào {personal_info.full_name}"
            position_statement = f"bạn đang muốn ứng tuyển vào vị trí {job_position}"
            assessment_statement = self._generate_assessment_statement(assessment_level)
            reason_statement = self._generate_reason_statement(assessment_level)
            
            # Tạo câu đánh giá hoàn chỉnh
            full_assessment = f"{greeting}, {position_statement}. {assessment_statement} {reason_statement}."
            
            logger.info(f"✅ Generated simple feedback for {personal_info.full_name}")
            
            return {
                "personalized_assessment": full_assessment,
                "assessment_level": assessment_level,
                "overall_score": overall_score
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating personalized feedback: {e}")
            return {
                "personalized_assessment": "Xin chào ứng viên, tôi đã phân tích CV của bạn.",
                "assessment_level": "unknown",
                "overall_score": 0
            }
    
    def _get_assessment_level(self, score: float) -> str:
        """Xác định level đánh giá dựa trên score"""
        if score >= 85:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"
    
    def _generate_assessment_statement(self, level: str) -> str:
        """Tạo câu đánh giá"""
        import random
        templates = self.assessment_templates.get(level, self.assessment_templates["fair"])
        return random.choice(templates)
    
    def _generate_reason_statement(self, level: str) -> str:
        """Tạo câu giải thích lý do"""
        import random
        templates = self.reason_templates.get(level, self.reason_templates["fair"])
        return random.choice(templates)
