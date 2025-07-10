import os
import json
import re
import random
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter

from .cv_parser import CVParser
from .cv_quality_analyzer import CVQualityAnalyzer
from .suggestion_generator import SuggestionGenerator
from ..models.shared_models import ParsedCV
from ..data.evaluate_cv import evaluate_cv, extract_sections_from_text, extract_entities_from_sections

class CVEvaluationService:
    """
    Service tích hợp đánh giá CV theo yêu cầu:
    1. Import CV và chọn nghề ứng tuyển
    2. Đánh giá CV theo tiêu chí nghề đó
    3. Tính điểm ATS (40% nếu CV chuẩn)
    4. Đánh giá linh hoạt, tự nhiên như con người
    5. Sử dụng mô hình đã train để học pattern CV tốt
    """
    
    def __init__(self):
        self.cv_parser = CVParser()
        self.quality_analyzer = CVQualityAnalyzer()
        self.suggestion_generator = SuggestionGenerator()
        
        # Load mô hình đã train (nếu có)
        self.ml_model = None
        self.vectorizer = None
        self.feature_importance = None
        self._load_trained_model()
        
        # Bộ từ khóa section (từ evaluate_cv.py)
        self.section_keywords = [
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
        
        # Stopwords
        self.stopwords = {
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
        
        # Đường dẫn file tiêu chí
        self.criteria_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'cv_criteria_final.json')
        
        # Template feedback linh hoạt
        self.feedback_templates = {
            "Xuất sắc": [
                "CV của bạn rất ấn tượng và phù hợp với vị trí này!",
                "Bạn đã trình bày đầy đủ các kỹ năng và kinh nghiệm cần thiết. Tuyệt vời!",
                "Đây là một CV rất chuyên nghiệp, thể hiện rõ năng lực và sự phù hợp.",
                "CV của bạn cho thấy bạn có đủ điều kiện và kinh nghiệm cho vị trí này."
            ],
            "Tốt": [
                "CV của bạn khá tốt, chỉ cần bổ sung thêm một số chi tiết nhỏ.",
                "Bạn đã có nền tảng vững chắc, hãy nhấn mạnh thêm các kỹ năng nổi bật.",
                "CV của bạn thể hiện sự phù hợp tốt, chỉ cần hoàn thiện thêm một chút.",
                "Bạn có tiềm năng tốt cho vị trí này, hãy làm nổi bật thêm kinh nghiệm."
            ],
            "Trung bình": [
                "CV của bạn còn thiếu một số kỹ năng quan trọng, hãy bổ sung để tăng cơ hội.",
                "Bạn nên mô tả chi tiết hơn về kinh nghiệm và kỹ năng chuyên môn.",
                "CV của bạn cần được cải thiện để phù hợp hơn với yêu cầu công việc.",
                "Bạn có thể tăng cơ hội bằng cách bổ sung thêm các kỹ năng cần thiết."
            ],
            "Cần cải thiện": [
                "CV của bạn cần được hoàn thiện thêm để phù hợp với vị trí này.",
                "Hãy bổ sung các kỹ năng, kinh nghiệm liên quan và trình bày rõ ràng hơn.",
                "CV của bạn cần được cải thiện đáng kể để đáp ứng yêu cầu công việc.",
                "Bạn nên xem xét lại và bổ sung thêm các thông tin cần thiết."
            ]
        }
    
    def _load_trained_model(self):
        """Load mô hình đã train và trích xuất feature importance"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cv_job_classifier.pkl')
            if os.path.exists(model_path):
                self.ml_model = joblib.load(model_path)
                print("✅ Đã load mô hình đã train thành công")
                
                # Trích xuất feature importance từ Random Forest
                if hasattr(self.ml_model, 'feature_importances_'):
                    self.feature_importance = self.ml_model.feature_importances_
                    print("✅ Đã trích xuất feature importance từ mô hình")
                else:
                    print("⚠️ Mô hình không có feature importance")
            else:
                print("⚠️ Không tìm thấy file mô hình đã train")
        except Exception as e:
            print(f"❌ Lỗi khi load mô hình: {e}")
    
    def _get_important_features_for_job(self, job_category: str) -> List[str]:
        """Lấy các feature quan trọng cho ngành nghề cụ thể từ mô hình đã train"""
        if self.ml_model is None or self.feature_importance is None:
            return []
        
        try:
            # Lấy top features quan trọng nhất
            top_indices = np.argsort(self.feature_importance)[-20:]  # Top 20 features
            important_features = []
            
            # Nếu có vectorizer, lấy tên feature
            if hasattr(self.ml_model, 'feature_names_in_'):
                for idx in top_indices:
                    feature_name = self.ml_model.feature_names_in_[idx]
                    if feature_name not in self.stopwords and len(feature_name) > 2:
                        important_features.append(feature_name)
            
            return important_features[:10]  # Trả về top 10
        except Exception as e:
            print(f"❌ Lỗi khi lấy important features: {e}")
            return []
    
    def _analyze_cv_with_ml_insights(self, cv_text: str, job_category: str) -> Dict:
        """Phân tích CV với insights từ mô hình ML đã train"""
        ml_insights = {
            'predicted_job': None,
            'confidence': 0.0,
            'important_features_missing': [],
            'important_features_found': [],
            'ml_suggestions': []
        }
        
        if self.ml_model is not None:
            try:
                # Dự đoán ngành nghề
                cv_vector = self.vectorizer.transform([cv_text]) if self.vectorizer else None
                if cv_vector is not None:
                    prediction = self.ml_model.predict(cv_vector)[0]
                    confidence = np.max(self.ml_model.predict_proba(cv_vector))
                    
                    ml_insights['predicted_job'] = prediction
                    ml_insights['confidence'] = confidence
                    
                    # So sánh với ngành user chọn
                    if prediction != job_category:
                        ml_insights['ml_suggestions'].append(
                            f"ML gợi ý ngành: {prediction} (độ tin cậy: {confidence:.2f})"
                        )
                
                # Lấy important features cho ngành này
                important_features = self._get_important_features_for_job(job_category)
                cv_lower = cv_text.lower()
                
                for feature in important_features:
                    if feature.lower() in cv_lower:
                        ml_insights['important_features_found'].append(feature)
                    else:
                        ml_insights['important_features_missing'].append(feature)
                
                # Tạo gợi ý từ ML insights
                if ml_insights['important_features_missing']:
                    ml_insights['ml_suggestions'].append(
                        f"Nên bổ sung: {', '.join(ml_insights['important_features_missing'][:5])}"
                    )
                
            except Exception as e:
                print(f"❌ Lỗi khi phân tích với ML: {e}")
        
        return ml_insights
    
    def evaluate_cv_comprehensive(self, cv_text: str, job_category: str) -> Dict:
        """
        Đánh giá CV toàn diện theo yêu cầu:
        1. Đánh giá theo tiêu chí nghề nghiệp
        2. Tính điểm ATS (40% nếu CV chuẩn)
        3. Đánh giá linh hoạt, tự nhiên
        4. Sử dụng insights từ mô hình ML đã train
        """
        
        # Bước 1: Đánh giá CV theo tiêu chí nghề nghiệp
        job_evaluation = evaluate_cv(cv_text, job_category)
        
        # Bước 2: Parse CV để phân tích cấu trúc
        parsed_cv = self.cv_parser.parse(cv_text)
        quality_analysis = self.quality_analyzer.analyze(parsed_cv)
        
        # Bước 3: Phân tích với insights từ ML
        ml_insights = self._analyze_cv_with_ml_insights(cv_text, job_category)
        
        # Bước 4: Tính điểm ATS
        ats_score = self._calculate_ats_score(job_evaluation, quality_analysis)
        
        # Bước 5: Tạo feedback linh hoạt
        flexible_feedback = self._generate_flexible_feedback(job_evaluation, parsed_cv, ml_insights)
        
        # Bước 6: Tạo gợi ý cải thiện
        suggestions = self._generate_improvement_suggestions(job_evaluation, parsed_cv, ml_insights)
        
        return {
            "job_category": job_category,
            "ats_score": ats_score,
            "level": job_evaluation.get('level', 'Cần cải thiện'),
            "overall_feedback": flexible_feedback,
            "found_entities": job_evaluation.get('found_entities', []),
            "improvement_suggestions": suggestions,
            "quality_score": quality_analysis.get('quality_score', 0),
            "strengths": quality_analysis.get('strengths', []),
            "job_evaluation_score": job_evaluation.get('total_score', 0),
            "ml_insights": ml_insights
        }
    
    def _calculate_ats_score(self, job_evaluation: Dict, quality_analysis: Dict) -> int:
        """
        Tính điểm ATS:
        - 40% nếu CV đạt chuẩn (đủ entity bắt buộc và cấu trúc tốt)
        - Điểm dựa trên chất lượng CV và entity matching
        """
        job_score = job_evaluation.get('total_score', 0) / 100  # Chuyển về thang 0-1
        quality_score = quality_analysis.get('quality_score', 0)
        
        # Tính điểm ATS (40% tối đa)
        ats_score = 0
        
        # Nếu CV đạt chuẩn (job_score >= 0.8 và quality_score >= 0.75)
        if job_score >= 0.8 and quality_score >= 0.75:
            ats_score = 40
        elif job_score >= 0.6 and quality_score >= 0.5:
            ats_score = int(30 * job_score)  # 0-30 điểm
        elif job_score >= 0.4 and quality_score >= 0.25:
            ats_score = int(20 * job_score)  # 0-20 điểm
        else:
            ats_score = int(10 * job_score)  # 0-10 điểm
        
        return ats_score
    
    def _generate_flexible_feedback(self, job_evaluation: Dict, parsed_cv: ParsedCV, ml_insights: Dict) -> str:
        """
        Tạo feedback linh hoạt, tự nhiên như con người với insights từ ML
        """
        level = job_evaluation.get('level', 'Cần cải thiện')
        found_entities = job_evaluation.get('found_entities', [])
        
        # Chọn template feedback ngẫu nhiên
        template = random.choice(self.feedback_templates.get(level, self.feedback_templates["Cần cải thiện"]))
        
        # Thêm thông tin cụ thể về ưu điểm
        strengths = []
        if found_entities:
            strengths.append(f"Ưu điểm: {', '.join(found_entities[:3])}")  # Giới hạn 3 entity
        
        if parsed_cv.skills:
            strengths.append(f"Kỹ năng: {', '.join(parsed_cv.skills[:3])}")
        
        if parsed_cv.experience:
            strengths.append("Có kinh nghiệm làm việc")
        
        if parsed_cv.education:
            strengths.append("Có trình độ học vấn")
        
        # Thêm insights từ ML
        if ml_insights.get('important_features_found'):
            strengths.append(f"Từ khóa quan trọng: {', '.join(ml_insights['important_features_found'][:3])}")
        
        # Thêm nhược điểm
        weaknesses = []
        if not found_entities:
            weaknesses.append("Thiếu các kỹ năng chuyên môn quan trọng")
        
        if not parsed_cv.skills:
            weaknesses.append("Chưa nêu rõ kỹ năng")
        
        if not parsed_cv.experience:
            weaknesses.append("Thiếu kinh nghiệm làm việc")
        
        # Thêm nhược điểm từ ML insights
        if ml_insights.get('important_features_missing'):
            weaknesses.append(f"Thiếu từ khóa quan trọng: {', '.join(ml_insights['important_features_missing'][:3])}")
        
        # Tạo feedback hoàn chỉnh
        feedback = template
        
        if strengths:
            feedback += f" {', '.join(strengths)}."
        
        if weaknesses:
            feedback += f" Nhược điểm: {', '.join(weaknesses)}."
        
        # Thêm gợi ý từ ML nếu có
        if ml_insights.get('ml_suggestions'):
            feedback += f" Gợi ý từ AI: {ml_insights['ml_suggestions'][0]}"
        
        return feedback
    
    def _generate_improvement_suggestions(self, job_evaluation: Dict, parsed_cv: ParsedCV, ml_insights: Dict) -> List[str]:
        """
        Tạo gợi ý cải thiện cụ thể với insights từ ML
        """
        suggestions = []
        
        # Gợi ý dựa trên entity thiếu
        if job_evaluation.get('total_score', 0) < 80:
            suggestions.append("Hãy bổ sung thêm các kỹ năng chuyên môn liên quan đến ngành nghề.")
            suggestions.append("Mô tả chi tiết hơn về kinh nghiệm làm việc.")
            suggestions.append("Thêm các chứng chỉ hoặc bằng cấp liên quan.")
        
        # Gợi ý dựa trên cấu trúc CV
        if not parsed_cv.skills:
            suggestions.append("Bổ sung mục 'Kỹ năng' với các kỹ năng chuyên môn.")
        
        if not parsed_cv.experience:
            suggestions.append("Thêm mục 'Kinh nghiệm làm việc' với các vị trí đã từng đảm nhiệm.")
        
        if not parsed_cv.education:
            suggestions.append("Bổ sung thông tin về học vấn và bằng cấp.")
        
        if not parsed_cv.summary:
            suggestions.append("Thêm phần tóm tắt hoặc mục tiêu nghề nghiệp.")
        
        # Gợi ý từ ML insights
        if ml_insights.get('important_features_missing'):
            suggestions.append(f"Bổ sung các từ khóa quan trọng: {', '.join(ml_insights['important_features_missing'][:5])}")
        
        if ml_insights.get('predicted_job') and ml_insights.get('predicted_job') != job_evaluation.get('job_category'):
            suggestions.append(f"Xem xét ngành nghề: {ml_insights['predicted_job']} (AI gợi ý)")
        
        # Gợi ý chung
        if not suggestions:
            suggestions.append("CV của bạn khá tốt, hãy tự tin ứng tuyển!")
        else:
            suggestions.append("Hãy tùy chỉnh CV để nhấn mạnh sự phù hợp với từng vị trí cụ thể.")
        
        return suggestions

# Test function
if __name__ == "__main__":
    service = CVEvaluationService()
    
    # Test với CV mẫu
    sample_cv = """
    NGUYỄN VĂN A
    Email: nva@email.com | Phone: 0123456789

    MỤC TIÊU NGHỀ NGHIỆP
    Trở thành một kế toán viên chuyên nghiệp với kinh nghiệm trong lĩnh vực tài chính.

    KINH NGHIỆM LÀM VIỆC
    Công ty ABC (01/2020 - Hiện tại)
    Kế toán viên
    - Lập báo cáo tài chính hàng tháng
    - Quản lý sổ sách kế toán
    - Xử lý các giao dịch tài chính

    HỌC VẤN
    Đại học Kinh tế (2016 - 2020)
    Chuyên ngành: Kế toán

    KỸ NĂNG
    - Excel, Word, PowerPoint
    - Phần mềm kế toán
    - Báo cáo tài chính
    - Quản lý sổ sách
    """
    
    result = service.evaluate_cv_comprehensive(sample_cv, "ACCOUNTANT")
    print(json.dumps(result, ensure_ascii=False, indent=2)) 