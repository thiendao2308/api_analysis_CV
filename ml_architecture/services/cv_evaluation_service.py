import os
import json
import re
import random
import joblib
import numpy as np
import spacy
from typing import Dict, List, Tuple, Optional
from collections import Counter

# Fix relative imports
try:
    from .cv_parser import IntelligentCVParser
    from .cv_quality_analyzer import CVQualityAnalyzer
    from .suggestion_generator import SuggestionGenerator
    from ..models.shared_models import ParsedCV
    from ..data.evaluate_cv import evaluate_cv, extract_sections_from_text, extract_entities_from_sections
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from services.cv_parser import IntelligentCVParser
    from services.cv_quality_analyzer import CVQualityAnalyzer
    from services.suggestion_generator import SuggestionGenerator
    from models.shared_models import ParsedCV
    from data.evaluate_cv import evaluate_cv, extract_sections_from_text, extract_entities_from_sections

class CVEvaluationService:
    """
    Service tích hợp đánh giá CV theo yêu cầu:
    BƯỚC 3: So sánh CV-JD để tính độ phù hợp (MML)
    BƯỚC 5: Liệt kê kỹ năng còn thiếu (MML)  
    BƯỚC 6: Chấm điểm tổng thể ATS (MML)
    """
    
    def __init__(self):
        self.cv_parser = IntelligentCVParser()
        self.quality_analyzer = CVQualityAnalyzer()
        self.suggestion_generator = SuggestionGenerator()
        
        # Load mô hình đã train (nếu có)
        self.ml_model = None
        self.vectorizer = None
        self.feature_importance = None
        self._load_trained_model()
        
        # Load NER model cho JD analysis
        self.jd_nlp = self._load_jd_ner_model()
        
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
    
    def _load_jd_ner_model(self):
        """Load NER model cho JD analysis - BƯỚC 2"""
        try:
            # Thử load model mới đã train
            model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_full', 'model-best')
            print(f"BƯỚC 2: Đang tải JD NER model từ: {model_path}")
            return spacy.load(model_path)
        except OSError:
            try:
                # Fallback sang model cũ
                model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model', 'model-best')
                print(f"BƯỚC 2: Fallback sang JD NER model cũ: {model_path}")
                return spacy.load(model_path)
            except OSError:
                print("BƯỚC 2: Không tìm thấy JD NER model, sử dụng model mặc định")
                return None
    
    def extract_jd_skills(self, jd_text: str) -> List[str]:
        """BƯỚC 2: Trích xuất skills từ JD sử dụng NER model"""
        if not self.jd_nlp:
            return []
        
        try:
            doc = self.jd_nlp(jd_text)
            skills = []
            
            for ent in doc.ents:
                if ent.label_ == "SKILL":
                    skill_text = ent.text.strip()
                    if skill_text and len(skill_text) > 1:
                        skills.append(skill_text)
            
            return list(set(skills))  # Loại bỏ duplicates
        except Exception as e:
            print(f"BƯỚC 2: Lỗi khi trích xuất skills từ JD: {e}")
            return []
    
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
            # Đây là logic đơn giản, có thể mở rộng dựa trên feature importance
            important_features = []
            
            # Thêm các từ khóa quan trọng cho từng ngành
            job_keywords = {
                "INFORMATION-TECHNOLOGY": ["python", "java", "javascript", "react", "node.js", "sql", "aws", "docker"],
                "ENGINEERING": ["autocad", "solidworks", "matlab", "engineering", "design", "analysis"],
                "FINANCE": ["excel", "financial", "accounting", "budget", "analysis", "reporting"],
                "SALES": ["sales", "customer", "negotiation", "communication", "target", "revenue"],
                "HR": ["recruitment", "hiring", "employee", "training", "policy", "benefits"],
                "MARKETING": ["marketing", "social media", "campaign", "brand", "content", "analytics"]
            }
            
            if job_category.upper() in job_keywords:
                important_features = job_keywords[job_category.upper()]
            
            return important_features
        except Exception as e:
            print(f"❌ Lỗi khi lấy important features: {e}")
            return []
    
    def _analyze_cv_with_ml_insights(self, cv_text: str, job_category: str) -> Dict:
        """BƯỚC 3: Phân tích CV với ML insights"""
        ml_insights = {
            'ml_score': 0,
            'ml_suggestions': [],
            'important_features': []
        }
        
        try:
            # Lấy important features cho job category
            important_features = self._get_important_features_for_job(job_category)
            ml_insights['important_features'] = important_features
            
            # Phân tích matching với important features
            cv_lower = cv_text.lower()
            matched_features = []
            
            for feature in important_features:
                if feature.lower() in cv_lower:
                    matched_features.append(feature)
            
            # Tính điểm ML
            if important_features:
                ml_score = len(matched_features) / len(important_features) * 100
                ml_insights['ml_score'] = ml_score
            
            # Tạo gợi ý ML
            if matched_features:
                ml_insights['ml_suggestions'].append(
                    f"✅ Bạn đã có các kỹ năng quan trọng: {', '.join(matched_features)}"
                )
            
            missing_features = [f for f in important_features if f.lower() not in cv_lower]
            if missing_features:
                ml_insights['ml_suggestions'].append(
                    f"⚠️ Cần bổ sung: {', '.join(missing_features[:3])}"
                )
            
        except Exception as e:
            print(f"❌ Lỗi khi phân tích ML insights: {e}")
        
        return ml_insights
    
    @staticmethod
    def _normalize_skill(skill):
        return skill.strip().lower()

    def _extract_skills_hybrid(self, text, job_category=None, is_cv=True):
        # ML extraction
        if is_cv:
            skills_ml = self.cv_parser.extract_skills(text, job_category)
        else:
            skills_ml = self.extract_jd_skills(text)
        # LLM extraction
        skills_llm = []
        # Nếu là JD, gọi OpenAI API để trích xuất skills (nếu có API key)
        if not is_cv:
            try:
                from ml_architecture.services.llm_api_extractor_jd import extract_skills_from_jd
                skills_llm_str = extract_skills_from_jd(text)
                if isinstance(skills_llm_str, str):
                    skills_llm = [s.strip() for s in skills_llm_str.split(",") if s.strip()]
            except Exception as e:
                print(f"[Hybrid JD] Lỗi khi gọi OpenAI API: {e}")
        # Union, loại trùng, chuẩn hóa
        all_skills = set(self._normalize_skill(s) for s in skills_ml) | set(self._normalize_skill(s) for s in skills_llm)
        # Trả về dạng chuẩn hóa (capitalize)
        return {
            "skills_ml": sorted(set(skills_ml)),
            "skills_llm": sorted(set(skills_llm)),
            "skills_union": sorted(s.capitalize() for s in all_skills if s)
        }

    def evaluate_cv_comprehensive(self, cv_text: str, job_category: str, job_position: str = None, jd_text: str = None, job_requirements: str = None) -> Dict:
        """
        BƯỚC 3: So sánh CV-JD để tính độ phù hợp (MML)
        BƯỚC 5: Liệt kê kỹ năng còn thiếu (MML)  
        BƯỚC 6: Chấm điểm tổng thể ATS (MML)
        """
        try:
            print(f"🚀 BƯỚC 3: Bắt đầu phân tích CV-JD cho {job_category} - {job_position}")
            
            # BƯỚC 1: Parse CV với parser thông minh
            from .cv_parser import parse_cv_file
            import tempfile
            import os
            
            # Tạo file tạm để parse
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(cv_text)
                temp_file_path = f.name
            
            try:
                parsed_cv = parse_cv_file(temp_file_path, job_category)
                print(f"✅ BƯỚC 1: Parse CV thành công - Job Title: {parsed_cv.get('job_title', 'N/A')}")
            finally:
                # Xóa file tạm
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
            # BƯỚC 2: Phân tích JD và trích xuất skills (hybrid)
            jd_skills_hybrid = {"skills_ml": [], "skills_llm": [], "skills_union": []}
            if jd_text:
                jd_skills_hybrid = self._extract_skills_hybrid(jd_text, job_category, is_cv=False)
                print(f"✅ BƯỚC 2: Trích xuất {len(jd_skills_hybrid['skills_union'])} skills (hybrid) từ JD")
            # BƯỚC 3: Trích xuất skills từ CV (hybrid)
            cv_skills_hybrid = self._extract_skills_hybrid(cv_text, job_category, is_cv=True)
            print(f"✅ BƯỚC 3: Trích xuất {len(cv_skills_hybrid['skills_union'])} skills (hybrid) từ CV")
            # BƯỚC 4: So sánh CV-JD
            cv_skills = cv_skills_hybrid['skills_union']
            jd_skills = jd_skills_hybrid['skills_union']
            cv_job_title = parsed_cv.get('job_title', '')
            position_match_score = self._check_position_match(cv_job_title, job_position, job_category)
            matching_skills = [skill for skill in cv_skills if self._normalize_skill(skill) in [self._normalize_skill(jd_skill) for jd_skill in jd_skills]]
            missing_skills = [skill for skill in jd_skills if self._normalize_skill(skill) not in [self._normalize_skill(cv_skill) for cv_skill in cv_skills]]
            skills_match_score = len(matching_skills) / max(len(jd_skills), 1) * 100 if jd_skills else 0
            print(f"✅ BƯỚC 4: Skills matching - {len(matching_skills)}/{len(jd_skills)} ({skills_match_score:.1f}%)")
            # BƯỚC 5: Phân tích chất lượng CV
            from ..models.shared_models import ParsedCV
            parsed_cv_obj = ParsedCV(
                summary=parsed_cv.get('sections', {}).get('summary', ''),
                skills=cv_skills,
                experience=parsed_cv.get('sections', {}).get('experience', ''),
                education=parsed_cv.get('sections', {}).get('education', '')
            )
            quality_analysis = self.quality_analyzer.analyze(parsed_cv_obj)
            print(f"✅ BƯỚC 5: Phân tích chất lượng CV hoàn tất")
            # BƯỚC 6: ML insights
            ml_insights = self._analyze_cv_with_ml_insights(cv_text, job_category)
            print(f"✅ BƯỚC 6: ML insights hoàn tất")
            # BƯỚC 7: Tính điểm ATS và Overall
            ats_score = self._calculate_ats_score(quality_analysis, parsed_cv_obj)
            overall_score = self._calculate_overall_score(
                ats_score, quality_analysis, len(cv_skills), len(jd_skills),
                cv_skills, jd_skills, job_category, position_match_score
            )
            print(f"✅ BƯỚC 7: ATS Score: {ats_score}, Overall Score: {overall_score}")
            # BƯỚC 8: Tạo feedback và suggestions
            feedback = self._generate_flexible_feedback(quality_analysis, parsed_cv_obj, ml_insights, jd_skills, job_category, overall_score)
            suggestions = self._generate_improvement_suggestions(quality_analysis, parsed_cv_obj, ml_insights, jd_skills)
            # Tạo kết quả chi tiết
            result = {
                "cv_analysis": {
                    "job_title": cv_job_title,
                    "skills": cv_skills,
                    "skills_ml": cv_skills_hybrid['skills_ml'],
                    "skills_llm": cv_skills_hybrid['skills_llm'],
                    "experience": parsed_cv.get('experience', []),
                    "education": parsed_cv.get('education', []),
                    "projects": parsed_cv.get('projects', []),
                    "sections": parsed_cv.get('sections', {})
                },
                "jd_analysis": {
                    "extracted_skills": jd_skills,
                    "skills_ml": jd_skills_hybrid['skills_ml'],
                    "skills_llm": jd_skills_hybrid['skills_llm'],
                    "jd_text": jd_text
                },
                "matching_analysis": {
                    "matching_skills": matching_skills,
                    "missing_skills": missing_skills,
                    "skills_match_score": round(skills_match_score, 1),
                    "position_match_score": position_match_score,
                    "total_skills_cv": len(cv_skills),
                    "total_skills_jd": len(jd_skills)
                },
                "quality_analysis": quality_analysis,
                "ml_insights": ml_insights,
                "scores": {
                    "ats_score": ats_score,
                    "overall_score": overall_score
                },
                "feedback": feedback,
                "suggestions": suggestions,
                "job_category": job_category,
                "job_position": job_position
            }
            print(f"🎉 Phân tích hoàn tất - Overall Score: {overall_score}")
            return result
        except Exception as e:
            print(f"❌ Lỗi trong evaluate_cv_comprehensive: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"Lỗi phân tích: {str(e)}",
                "cv_analysis": {},
                "jd_analysis": {},
                "matching_analysis": {},
                "quality_analysis": {},
                "ml_insights": {},
                "scores": {"ats_score": 0, "overall_score": 0},
                "feedback": "Có lỗi xảy ra trong quá trình phân tích",
                "suggestions": ["Vui lòng thử lại với CV khác"],
                "job_category": job_category,
                "job_position": job_position
            }

    def _check_position_match(self, cv_job_title: str, job_position: str, job_category: str) -> int:
        """Kiểm tra sự phù hợp giữa job title trong CV và job position được chọn"""
        if not cv_job_title or not job_position:
            return 50  # Trung bình nếu không có thông tin
        
        cv_title_lower = cv_job_title.lower()
        job_pos_lower = job_position.lower()
        
        # Mapping job position values to keywords
        position_keywords = {
            "FRONTEND_DEVELOPER": ["frontend", "front-end", "front end", "react", "angular", "vue"],
            "BACKEND_DEVELOPER": ["backend", "back-end", "back end", "python", "java", "node"],
            "FULLSTACK_DEVELOPER": ["fullstack", "full-stack", "full stack", "fullstack"],
            "MOBILE_DEVELOPER": ["mobile", "android", "ios", "flutter", "react native"],
            "DATA_SCIENTIST": ["data scientist", "data science", "machine learning", "ai"],
            "DEVOPS_ENGINEER": ["devops", "cloud", "aws", "azure", "docker"],
            "QA_ENGINEER": ["qa", "quality", "test", "testing"],
            "UI_UX_DESIGNER": ["ui", "ux", "designer", "user interface"],
            "SEO_SPECIALIST": ["seo", "search engine", "digital marketing"],
            "DIGITAL_MARKETING": ["digital marketing", "marketing", "social media"],
            "SALES_REPRESENTATIVE": ["sales", "representative", "business development"],
            "HR_SPECIALIST": ["hr", "human resources", "recruitment"],
            "ACCOUNTANT": ["accountant", "accounting", "finance"],
            "FINANCIAL_ANALYST": ["financial analyst", "finance", "analysis"],
        }
        
        # Kiểm tra match
        if job_position in position_keywords:
            keywords = position_keywords[job_position]
            for keyword in keywords:
                if keyword in cv_title_lower:
                    return 100  # Perfect match
            return 30  # Low match
        else:
            # Fallback: kiểm tra từ khóa chung
            if any(word in cv_title_lower for word in job_pos_lower.split('_')):
                return 80
            return 40

    def _calculate_ats_score(self, quality_analysis: Dict, parsed_cv: ParsedCV) -> int:
        """BƯỚC 6: Tính điểm ATS dựa trên chất lượng CV"""
        ats_score = 0
        
        # Điểm cho format chuẩn
        if quality_analysis.get('quality_score', 0) >= 0.75:
            ats_score += 20
        
        # Điểm cho skills
        if parsed_cv.skills:
            ats_score += min(len(parsed_cv.skills) * 2, 20)  # Tối đa 20 điểm
        
        # Điểm cho experience
        if parsed_cv.experience:
            ats_score += 15
        
        # Điểm cho education
        if parsed_cv.education:
            ats_score += 10
        
        # Điểm cho summary
        if parsed_cv.summary:
            ats_score += 10
        
        return min(ats_score, 100)

    def _calculate_overall_score(self, ats_score: int, quality_analysis: Dict, cv_skills_count: int, jd_skills_count: int, cv_skills: List[str], jd_skills: List[str], job_category: str, position_match_score: int) -> int:
        """
        Tính điểm tổng thể với logic mới:
        - ATS Score: 40%
        - Skills Matching: 30%
        - Position Match: 20%
        - Quality Analysis: 10%
        """
        try:
            # 1. ATS Score (40%)
            ats_component = ats_score * 0.4
            
            # 2. Skills Matching (30%)
            skills_match = 0
            if jd_skills_count > 0:
                matching_count = len([s for s in cv_skills if s.lower() in [js.lower() for js in jd_skills]])
                skills_match = (matching_count / jd_skills_count) * 100
            skills_component = skills_match * 0.3
            
            # 3. Position Match (20%)
            position_component = position_match_score * 0.2
            
            # 4. Quality Analysis (10%)
            quality_score = quality_analysis.get('overall_score', 50)
            quality_component = quality_score * 0.1
            
            # Tính tổng
            overall_score = ats_component + skills_component + position_component + quality_component
            
            # Đảm bảo điểm trong khoảng 0-100
            overall_score = max(0, min(100, overall_score))
            
            print(f"📊 Overall Score Breakdown:")
            print(f"   - ATS Component: {ats_component:.1f}")
            print(f"   - Skills Component: {skills_component:.1f}")
            print(f"   - Position Component: {position_component:.1f}")
            print(f"   - Quality Component: {quality_component:.1f}")
            print(f"   - Total: {overall_score:.1f}")
            
            return round(overall_score)
            
        except Exception as e:
            print(f"❌ Lỗi tính overall score: {e}")
            return 50  # Fallback score

    def _generate_flexible_feedback(self, quality_analysis: Dict, parsed_cv: ParsedCV, ml_insights: Dict, jd_skills: List[str], job_category: str, overall_score: int = None) -> str:
        """BƯỚC 5: Tạo feedback linh hoạt dựa trên phân tích"""
        # Sử dụng overall_score đã được tính trước đó
        if overall_score is None:
            # Fallback: tính lại nếu cần
            ats_score = self._calculate_ats_score(quality_analysis, parsed_cv)
            overall_score = self._calculate_overall_score(
                ats_score,
                quality_analysis,
                len(parsed_cv.skills),
                len(jd_skills),
                parsed_cv.skills,
                jd_skills,
                job_category,
                50  # Default position match score
            )
        
        # Chọn template dựa trên điểm
        if overall_score >= 85:
            template_key = "Xuất sắc"
        elif overall_score >= 70:
            template_key = "Tốt"
        elif overall_score >= 50:
            template_key = "Trung bình"
        else:
            template_key = "Cần cải thiện"
        
        # Chọn feedback ngẫu nhiên từ template
        feedback = random.choice(self.feedback_templates[template_key])
        
        # Thêm thông tin cụ thể
        if jd_skills:
            cv_skills_set = set(parsed_cv.skills)
            jd_skills_set = set(jd_skills)
            matching_skills = cv_skills_set.intersection(jd_skills_set)
            
            if matching_skills:
                feedback += f"\n\n✅ Kỹ năng phù hợp: {', '.join(list(matching_skills)[:5])}"
            
            missing_skills = jd_skills_set - cv_skills_set
            if missing_skills:
                feedback += f"\n\n⚠️ Cần bổ sung: {', '.join(list(missing_skills)[:5])}"
        
        return feedback
    
    def _generate_improvement_suggestions(self, quality_analysis: Dict, parsed_cv: ParsedCV, ml_insights: Dict, jd_skills: List[str]) -> List[str]:
        """BƯỚC 4: Tạo gợi ý cải thiện"""
        suggestions = []
        
        # Gợi ý từ ML insights
        if ml_insights.get('ml_suggestions'):
            suggestions.extend(ml_insights['ml_suggestions'])
        
        # Gợi ý từ quality analysis
        if quality_analysis.get('quality_score', 0) < 0.75:
            suggestions.append("Cải thiện cấu trúc CV với các mục rõ ràng")
        
        # Gợi ý từ skills matching
        if jd_skills:
            cv_skills_set = set(parsed_cv.skills)
            jd_skills_set = set(jd_skills)
            missing_skills = jd_skills_set - cv_skills_set
            
            if missing_skills:
                suggestions.append(f"Bổ sung các kỹ năng: {', '.join(list(missing_skills)[:3])}")
        
        # Gợi ý từ parsed CV
        if not parsed_cv.summary:
            suggestions.append("Thêm phần tóm tắt/mục tiêu nghề nghiệp")
        
        if not parsed_cv.experience:
            suggestions.append("Bổ sung thông tin kinh nghiệm làm việc")
        
        return suggestions[:5]  # Giới hạn 5 gợi ý

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
    - Kế toán tài chính
    """
    
    sample_jd = """
    Tuyển dụng Kế toán viên
    Yêu cầu:
    - Kinh nghiệm kế toán
    - Thành thạo Excel
    - Biết sử dụng phần mềm kế toán
    """
    
    result = service.evaluate_cv_comprehensive(sample_cv, "FINANCE", sample_jd)
    print(f"Điểm tổng: {result['overall_score']}")
    print(f"Feedback: {result['feedback']}")
    print(f"Suggestions: {result['suggestions']}") 