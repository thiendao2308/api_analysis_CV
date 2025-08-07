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
    from .intelligent_jd_matcher import IntelligentJDMatcher
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
    from services.intelligent_jd_matcher import IntelligentJDMatcher
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
        # Lazy loading - chỉ khởi tạo các service nhẹ
        self.cv_parser = None
        self.quality_analyzer = None
        self.suggestion_generator = None
        
        # Load mô hình đã train (nếu có) - lazy loading
        self.ml_model = None
        self.vectorizer = None
        self.feature_importance = None
        
        # Load NER model cho JD analysis - lazy loading
        self.jd_nlp = None
        
        # Intelligent JD matching service
        self.intelligent_jd_matcher = None
        
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
        
        # LLM Feedback Generator - lazy loading
        self.llm_feedback_generator = None
        
        # Template feedback linh hoạt (fallback)
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
        if self.jd_nlp is not None:
            return self.jd_nlp
        try:
            # Thử load model mới đã train
            model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_full', 'model-best')
            print(f"BƯỚC 2: Đang tải JD NER model từ: {model_path}")
            self.jd_nlp = spacy.load(model_path)
            return self.jd_nlp
        except OSError:
            try:
                # Fallback sang model cũ
                model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model', 'model-best')
                print(f"BƯỚC 2: Fallback sang JD NER model cũ: {model_path}")
                self.jd_nlp = spacy.load(model_path)
                return self.jd_nlp
            except OSError:
                print("BƯỚC 2: Không tìm thấy JD NER model, sử dụng model mặc định")
                return None
    
    def extract_jd_skills(self, jd_text: str) -> List[str]:
        """BƯỚC 2: Trích xuất skills từ JD sử dụng NER model"""
        if self.jd_nlp is None:
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
        if self.ml_model is not None:
            return
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
            if self.cv_parser is None:
                from .cv_parser import IntelligentCVParser
                self.cv_parser = IntelligentCVParser()
            skills_ml = self.cv_parser.extract_skills(text, job_category)
        else:
            if self.jd_nlp is None:
                self._load_jd_ner_model() # Ensure JD NER model is loaded
            skills_ml = self.extract_jd_skills(text)
        
        # LLM extraction - Thêm cho cả CV và JD
        skills_llm = []
        try:
            if is_cv:
                # LLM extraction cho CV
                from ml_architecture.services.llm_api_extractor_cv import extract_skills_from_cv
                skills_llm_str = extract_skills_from_cv(text)
                if isinstance(skills_llm_str, str):
                    skills_llm = [s.strip() for s in skills_llm_str.split(",") if s.strip()]
            else:
                # LLM extraction cho JD
                from ml_architecture.services.llm_api_extractor_jd import extract_skills_from_jd
                skills_llm_str = extract_skills_from_jd(text)
                if isinstance(skills_llm_str, str):
                    skills_llm = [s.strip() for s in skills_llm_str.split(",") if s.strip()]
        except Exception as e:
            print(f"[Hybrid {'CV' if is_cv else 'JD'}] Lỗi khi gọi OpenAI API: {e}")
        
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
                # Cải thiện lấy job title: nếu None/rỗng, lấy từ LLM extraction nếu có
                job_title = parsed_cv.get('job_title', '')
                if not job_title:
                    # Lấy từ LLM extraction nếu có work experience
                    from ml_architecture.services.llm_api_extractor_cv import extract_cv_info_from_text
                    llm_cv_info = extract_cv_info_from_text(cv_text)
                    import json
                    try:
                        llm_cv_json = json.loads(llm_cv_info) if isinstance(llm_cv_info, str) else llm_cv_info
                        work_exp = llm_cv_json.get('Work experience') or llm_cv_json.get('work_experience')
                        if isinstance(work_exp, list) and len(work_exp) > 0:
                            # Ưu tiên vị trí gần nhất (mới nhất)
                            job_title = work_exp[0].get('position') or ''
                    except Exception:
                        pass
                parsed_cv['job_title'] = job_title
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
            # BƯỚC 4: Intelligent CV-JD matching
            cv_skills = cv_skills_hybrid['skills_union']
            jd_skills = jd_skills_hybrid['skills_union']
            cv_job_title = parsed_cv.get('job_title', '')
            position_match_score = self._check_position_match(cv_job_title, job_position, job_category)
            
            # Use intelligent JD matching
            if self.intelligent_jd_matcher is None:
                self.intelligent_jd_matcher = IntelligentJDMatcher()
            
            matching_result = self.intelligent_jd_matcher.intelligent_matching(cv_skills, jd_skills)
            matching_skills = matching_result['matching_skills']
            missing_skills = matching_result['missing_skills']
            skills_match_score = matching_result['match_score']
            
            print(f"✅ BƯỚC 4: Intelligent JD matching - {len(matching_skills)}/{len(jd_skills)} ({skills_match_score:.1f}%)")
            print(f"   - Exact matches: {len(matching_result.get('exact_matches', []))}")
            print(f"   - Family matches: {len(matching_result.get('family_matches', []))}")
            print(f"   - Semantic matches: {len(matching_result.get('semantic_matches', []))}")
            
            # Log family mapping details if available
            if 'family_mapping_details' in matching_result and matching_result['family_mapping_details']:
                print(f"   - Family mappings: {matching_result['family_mapping_details'][:3]}...")  # Show first 3
            # BƯỚC 5: Phân tích chất lượng CV
            from ..models.shared_models import ParsedCV
            parsed_cv_obj = ParsedCV(
                summary=parsed_cv.get('sections', {}).get('summary', ''),
                skills=cv_skills,
                experience=parsed_cv.get('sections', {}).get('experience', ''),
                education=parsed_cv.get('sections', {}).get('education', '')
            )
            if self.quality_analyzer is None:
                from .cv_quality_analyzer import CVQualityAnalyzer
                self.quality_analyzer = CVQualityAnalyzer()
            quality_analysis = self.quality_analyzer.analyze(parsed_cv_obj)
            print(f"✅ BƯỚC 5: Phân tích chất lượng CV hoàn tất")
            # BƯỚC 6: ML insights
            ml_insights = self._analyze_cv_with_ml_insights(cv_text, job_category)
            print(f"✅ BƯỚC 6: ML insights hoàn tất")
            # BƯỚC 7: Tính điểm ATS và Overall
            ats_score = self._calculate_ats_score(quality_analysis, parsed_cv_obj)
            overall_score = self._calculate_overall_score(
                ats_score, quality_analysis, len(cv_skills), len(jd_skills),
                cv_skills, jd_skills, job_category, position_match_score, matching_result
            )
            print(f"✅ BƯỚC 7: ATS Score: {ats_score}, Overall Score: {overall_score}")
            # BƯỚC 8: Tạo feedback thông minh bằng LLM
            llm_feedback = self._generate_intelligent_llm_feedback(
                cv_analysis={
                    "skills": cv_skills,
                    "experience": parsed_cv.get('experience', []),
                    "education": parsed_cv.get('education', [])
                },
                jd_analysis={
                    "extracted_skills": jd_skills
                },
                matching_analysis={
                    "matching_skills": matching_skills,
                    "missing_skills": missing_skills,
                    "skills_match_score": skills_match_score
                },
                quality_analysis=quality_analysis,
                overall_score=overall_score,
                job_category=job_category,
                job_position=job_position
            )
            
            # Fallback to traditional feedback if LLM fails
            if not llm_feedback:
                feedback = self._generate_flexible_feedback(quality_analysis, parsed_cv_obj, ml_insights, jd_skills, job_category, overall_score)
                suggestions = self._generate_improvement_suggestions(quality_analysis, parsed_cv_obj, ml_insights, jd_skills)
            else:
                feedback = llm_feedback.get('overall_assessment', '')
                suggestions = llm_feedback.get('specific_suggestions', [])
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
        
        # Mapping job position values to keywords với logic cải thiện
        position_keywords = {
            "FRONTEND_DEVELOPER": [
                "frontend", "front-end", "front end", "react", "angular", "vue",
                "fullstack", "full-stack", "full stack",  # Full-stack có thể làm Frontend
                "web developer", "ui developer", "client-side"
            ],
            "BACKEND_DEVELOPER": [
                "backend", "back-end", "back end", "python", "java", "node",
                "fullstack", "full-stack", "full stack",  # Full-stack có thể làm Backend
                "server-side", "api developer", "database"
            ],
            "FULLSTACK_DEVELOPER": [
                "fullstack", "full-stack", "full stack", "fullstack developer",
                "frontend", "backend", "web developer", "full stack developer"
            ],
            "MOBILE_DEVELOPER": [
                "mobile", "android", "ios", "flutter", "react native",
                "mobile developer", "app developer"
            ],
            "DATA_SCIENTIST": [
                "data scientist", "data science", "machine learning", "ai",
                "analytics", "statistics", "research"
            ],
            "DEVOPS_ENGINEER": [
                "devops", "cloud", "aws", "azure", "docker",
                "infrastructure", "system administrator", "platform engineer"
            ],
            "QA_ENGINEER": [
                "qa", "quality", "test", "testing", "quality assurance",
                "tester", "test engineer"
            ],
            "UI_UX_DESIGNER": [
                "ui", "ux", "designer", "user interface", "user experience",
                "web designer", "graphic designer"
            ],
            "SEO_SPECIALIST": [
                "seo", "search engine", "digital marketing", "optimization",
                "marketing specialist"
            ],
            "DIGITAL_MARKETING": [
                "digital marketing", "marketing", "social media",
                "content marketing", "online marketing"
            ],
            "SALES_REPRESENTATIVE": [
                "sales", "representative", "business development",
                "account manager", "sales executive"
            ],
            "HR_SPECIALIST": [
                "hr", "human resources", "recruitment", "talent acquisition",
                "personnel", "employee relations"
            ],
            "ACCOUNTANT": [
                "accountant", "accounting", "finance", "bookkeeper",
                "financial analyst"
            ],
            "FINANCIAL_ANALYST": [
                "financial analyst", "finance", "analysis", "investment",
                "financial planning"
            ],
        }
        
        # Kiểm tra match với logic cải thiện
        if job_position in position_keywords:
            keywords = position_keywords[job_position]
            
            # Kiểm tra exact match
            for keyword in keywords:
                if keyword in cv_title_lower:
                    return 100  # Perfect match
            
            # Kiểm tra fuzzy match cho lỗi chính tả
            from difflib import SequenceMatcher
            for keyword in keywords:
                similarity = SequenceMatcher(None, cv_title_lower, keyword).ratio()
                if similarity > 0.8:  # 80% similarity
                    return 90  # High match
            
            # Kiểm tra partial match
            for keyword in keywords:
                if any(word in cv_title_lower for word in keyword.split()):
                    return 70  # Good match
            
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

    def _calculate_overall_score(self, ats_score: int, quality_analysis: Dict, cv_skills_count: int, jd_skills_count: int, cv_skills: List[str], jd_skills: List[str], job_category: str, position_match_score: int, matching_result: Dict = None) -> int:
        """
        Tính điểm tổng thể với logic cải thiện:
        - ATS Score: 35%
        - Skills Matching: 30%
        - Position Match: 25%
        - Quality Analysis: 10%
        """
        try:
            # 1. ATS Score (35%)
            ats_component = ats_score * 0.35
            
            # 2. Skills Matching (30%) - Sử dụng kết quả từ Intelligent JD Matching
            skills_match = 0
            if jd_skills_count > 0:
                # Sử dụng matching_result từ Intelligent JD Matching
                if matching_result and 'match_score' in matching_result:
                    # Lấy match score từ intelligent matching
                    skills_match = matching_result.get('match_score', 0)
                else:
                    # Fallback: tính toán đơn giản
                    matching_skills = set(cv_skills) & set(jd_skills)
                    skills_match = (len(matching_skills) / jd_skills_count) * 100 if jd_skills_count > 0 else 0
            
            skills_component = skills_match * 0.30
            
            # 3. Position Match (25%) - Cải thiện với logic mới
            position_component = position_match_score * 0.25
            
            # 4. Quality Analysis (10%) - Sử dụng quality_score từ CV Quality Analyzer
            quality_score = quality_analysis.get('quality_score', 0) * 100  # Chuyển về thang 100
            quality_component = quality_score * 0.10
            
            # Tính tổng điểm
            overall_score = ats_component + skills_component + position_component + quality_component
            
            # Đảm bảo điểm không vượt quá 100
            overall_score = min(overall_score, 100)
            
            return int(overall_score)
            
        except Exception as e:
            print(f"❌ Lỗi tính overall score: {e}")
            return 50  # Điểm trung bình nếu có lỗi

    def _calculate_industry_specific_score(self, job_category: str, job_position: str, matching_result: dict, quality_score: float) -> dict:
        """Tính điểm theo ngành nghề cụ thể"""
        
        industry_weights = {
            "INFORMATION-TECHNOLOGY": {
                "technical_skills": 0.4,
                "soft_skills": 0.2,
                "experience": 0.25,
                "education": 0.15
            },
            "MARKETING": {
                "technical_skills": 0.3,
                "soft_skills": 0.35,
                "experience": 0.25,
                "education": 0.1
            },
            "FINANCE": {
                "technical_skills": 0.35,
                "soft_skills": 0.2,
                "experience": 0.3,
                "education": 0.15
            },
            "HUMAN-RESOURCES": {
                "technical_skills": 0.25,
                "soft_skills": 0.4,
                "experience": 0.25,
                "education": 0.1
            },
            "SALES": {
                "technical_skills": 0.2,
                "soft_skills": 0.45,
                "experience": 0.25,
                "education": 0.1
            },
            "HEALTHCARE": {
                "technical_skills": 0.4,
                "soft_skills": 0.25,
                "experience": 0.25,
                "education": 0.1
            },
            "EDUCATION": {
                "technical_skills": 0.25,
                "soft_skills": 0.35,
                "experience": 0.25,
                "education": 0.15
            },
            "DESIGN": {
                "technical_skills": 0.35,
                "soft_skills": 0.3,
                "experience": 0.25,
                "education": 0.1
            }
        }
        
        # Get weights for this industry
        weights = industry_weights.get(job_category.upper(), industry_weights["INFORMATION-TECHNOLOGY"])
        
        # Calculate component scores
        technical_score = matching_result.get('match_score', 0)
        soft_skills_score = min(100, quality_score * 1.2)  # Boost soft skills for quality
        experience_score = min(100, quality_score * 1.1)  # Experience based on quality
        education_score = min(100, quality_score * 0.9)  # Education slightly lower weight
        
        # Calculate weighted score
        weighted_score = (
            technical_score * weights["technical_skills"] +
            soft_skills_score * weights["soft_skills"] +
            experience_score * weights["experience"] +
            education_score * weights["education"]
        )
        
        return {
            "industry_score": weighted_score,
            "component_scores": {
                "technical_skills": technical_score,
                "soft_skills": soft_skills_score,
                "experience": experience_score,
                "education": education_score
            },
            "weights": weights,
            "industry": job_category
        }

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
    
    def _generate_intelligent_llm_feedback(self, cv_analysis: Dict, jd_analysis: Dict, matching_analysis: Dict, quality_analysis: Dict, overall_score: float, job_category: str, job_position: str) -> Dict:
        """Generate intelligent feedback using LLM"""
        try:
            # Lazy load LLM Feedback Generator
            if self.llm_feedback_generator is None:
                from .llm_feedback_generator import LLMFeedbackGenerator
                self.llm_feedback_generator = LLMFeedbackGenerator()
            
            # Generate intelligent feedback
            llm_feedback = self.llm_feedback_generator.generate_intelligent_feedback(
                cv_analysis=cv_analysis,
                jd_analysis=jd_analysis,
                matching_analysis=matching_analysis,
                quality_analysis=quality_analysis,
                overall_score=overall_score,
                job_category=job_category,
                job_position=job_position
            )
            
            print(f"✅ LLM Feedback generated successfully")
            return llm_feedback
            
        except Exception as e:
            print(f"❌ Error generating LLM feedback: {e}")
            return None
    
    def _generate_improvement_suggestions(self, quality_analysis: Dict, parsed_cv: ParsedCV, ml_insights: Dict, jd_skills: List[str]) -> List[str]:
        """BƯỚC 6: Tạo gợi ý cải thiện CV"""
        if self.suggestion_generator is None:
            from .suggestion_generator import SuggestionGenerator
            self.suggestion_generator = SuggestionGenerator()
            
        try:
            # Extract missing skills
            cv_skills_set = set(parsed_cv.skills)
            jd_skills_set = set(jd_skills)
            missing_skills = list(jd_skills_set - cv_skills_set)[:5]  # Limit to 5 missing skills
            
            # Extract matched skills
            matched_skills = list(cv_skills_set & jd_skills_set)[:5]  # Limit to 5 matched skills
            
            # Get job category from ml_insights if available
            job_category = ml_insights.get('important_features', [])
            if job_category:
                job_category = "INFORMATION-TECHNOLOGY"  # Default category
            
            suggestions = self.suggestion_generator.generate(
                matched_keywords=matched_skills,
                missing_keywords=missing_skills,
                cv_quality=quality_analysis,
                job_category=job_category
            )
            return suggestions
        except Exception as e:
            print(f"❌ Lỗi khi tạo gợi ý cải thiện: {e}")
            return ["Cần cải thiện CV để phù hợp hơn với yêu cầu công việc"]

    def _analyze_skill_gaps(self, cv_skills: List[str], jd_skills: List[str], job_category: str) -> dict:
        """Phân tích skill gaps chi tiết"""
        
        # Get missing skills
        missing_skills = [skill for skill in jd_skills if skill not in cv_skills]
        
        # Categorize missing skills by priority
        skill_priorities = {
            "INFORMATION-TECHNOLOGY": {
                "critical": ["JavaScript", "Python", "SQL", "Git", "React", "Node.js", "AWS", "Docker"],
                "important": ["TypeScript", "Vue.js", "MongoDB", "PostgreSQL", "Docker", "Kubernetes"],
                "nice_to_have": ["GraphQL", "Redis", "Elasticsearch", "Machine Learning"]
            },
            "MARKETING": {
                "critical": ["Google Ads", "Facebook Ads", "SEO", "Content Marketing", "Analytics"],
                "important": ["Email Marketing", "Social Media", "PPC", "Conversion Optimization"],
                "nice_to_have": ["Marketing Automation", "Influencer Marketing", "Video Marketing"]
            },
            "FINANCE": {
                "critical": ["Excel", "Financial Modeling", "Accounting", "Analysis"],
                "important": ["SAP", "Oracle", "QuickBooks", "Risk Management"],
                "nice_to_have": ["Python", "R", "Tableau", "Power BI"]
            },
            "HUMAN-RESOURCES": {
                "critical": ["Recruitment", "Employee Relations", "HR Policies", "Compliance"],
                "important": ["HRIS", "Performance Management", "Training", "Benefits"],
                "nice_to_have": ["HR Analytics", "Diversity & Inclusion", "Organizational Development"]
            },
            "SALES": {
                "critical": ["CRM", "Sales Techniques", "Negotiation", "Lead Generation"],
                "important": ["Salesforce", "Pipeline Management", "Customer Success"],
                "nice_to_have": ["Sales Analytics", "Social Selling", "Account Management"]
            },
            "HEALTHCARE": {
                "critical": ["Patient Care", "Clinical Skills", "Medical Records", "Compliance"],
                "important": ["EMR Systems", "Medical Terminology", "Patient Assessment"],
                "nice_to_have": ["Telemedicine", "Healthcare Analytics", "Quality Improvement"]
            },
            "EDUCATION": {
                "critical": ["Teaching", "Curriculum Development", "Student Assessment", "Classroom Management"],
                "important": ["LMS", "Educational Technology", "Lesson Planning"],
                "nice_to_have": ["Online Teaching", "Educational Research", "Special Education"]
            },
            "DESIGN": {
                "critical": ["Adobe Creative Suite", "UI/UX Design", "Visual Design", "Typography"],
                "important": ["Figma", "Sketch", "Prototyping", "User Research"],
                "nice_to_have": ["Motion Graphics", "3D Design", "Brand Strategy"]
            }
        }
        
        priorities = skill_priorities.get(job_category.upper(), skill_priorities["INFORMATION-TECHNOLOGY"])
        
        # Categorize missing skills
        critical_gaps = [skill for skill in missing_skills if skill in priorities["critical"]]
        important_gaps = [skill for skill in missing_skills if skill in priorities["important"]]
        nice_to_have_gaps = [skill for skill in missing_skills if skill in priorities["nice_to_have"]]
        other_gaps = [skill for skill in missing_skills if skill not in priorities["critical"] + priorities["important"] + priorities["nice_to_have"]]
        
        # Calculate gap severity
        total_critical = len(priorities["critical"])
        total_important = len(priorities["important"])
        
        critical_severity = len(critical_gaps) / total_critical if total_critical > 0 else 0
        important_severity = len(important_gaps) / total_important if total_important > 0 else 0
        
        overall_gap_severity = (critical_severity * 0.6) + (important_severity * 0.4)
        
        return {
            "critical_gaps": critical_gaps,
            "important_gaps": important_gaps,
            "nice_to_have_gaps": nice_to_have_gaps,
            "other_gaps": other_gaps,
            "gap_severity": {
                "critical": critical_severity,
                "important": important_severity,
                "overall": overall_gap_severity
            },
            "total_missing": len(missing_skills),
            "total_required": len(jd_skills)
        }

    def _generate_comprehensive_report(self, analysis_result: dict, job_category: str, job_position: str) -> dict:
        """Tạo comprehensive analysis report với industry insights"""
        
        # Extract key metrics
        overall_score = analysis_result.get('overall_score', 0)
        ats_score = analysis_result.get('ats_score', 0)
        skills_match_score = analysis_result.get('skills_match_score', 0)
        quality_score = analysis_result.get('quality_score', 0)
        
        # Industry-specific insights
        industry_insights = {
            "INFORMATION-TECHNOLOGY": {
                "market_demand": "Cao",
                "salary_range": "15-70M+ VND",
                "growth_rate": "15-20% annually",
                "key_trends": ["Cloud Computing", "AI/ML", "DevOps", "Cybersecurity"],
                "learning_path": "Online courses → Certifications → Projects → Open Source",
                "career_progression": "Junior → Mid → Senior → Lead → Architect"
            },
            "MARKETING": {
                "market_demand": "Cao",
                "salary_range": "12-50M+ VND",
                "growth_rate": "12-18% annually",
                "key_trends": ["Digital Marketing", "Social Commerce", "Marketing Automation", "Data Analytics"],
                "learning_path": "Certifications → Campaigns → Case Studies → Portfolio",
                "career_progression": "Assistant → Specialist → Manager → Director"
            },
            "FINANCE": {
                "market_demand": "Ổn định",
                "salary_range": "15-60M+ VND",
                "growth_rate": "8-12% annually",
                "key_trends": ["Fintech", "ESG Investing", "Automation", "Risk Management"],
                "learning_path": "Certifications → Industry Experience → Specialization",
                "career_progression": "Analyst → Senior Analyst → Manager → Director"
            },
            "HUMAN-RESOURCES": {
                "market_demand": "Ổn định",
                "salary_range": "12-50M+ VND",
                "growth_rate": "10-15% annually",
                "key_trends": ["HR Tech", "Remote Work", "Employee Experience", "Diversity & Inclusion"],
                "learning_path": "HR Certifications → Industry Experience → Specialization",
                "career_progression": "Assistant → Specialist → Manager → Director"
            },
            "SALES": {
                "market_demand": "Cao",
                "salary_range": "15-80M+ VND",
                "growth_rate": "12-20% annually",
                "key_trends": ["Digital Sales", "Social Selling", "Customer Success", "Sales Analytics"],
                "learning_path": "Sales Training → Industry Knowledge → Networking",
                "career_progression": "Sales Rep → Senior Rep → Manager → Director"
            },
            "HEALTHCARE": {
                "market_demand": "Cao",
                "salary_range": "20-80M+ VND",
                "growth_rate": "15-25% annually",
                "key_trends": ["Telemedicine", "AI Diagnostics", "Personalized Medicine", "Digital Health"],
                "learning_path": "Medical Certifications → Continuing Education → Specialization",
                "career_progression": "Junior Staff → Senior Staff → Specialist → Director"
            },
            "EDUCATION": {
                "market_demand": "Ổn định",
                "salary_range": "15-60M+ VND",
                "growth_rate": "8-12% annually",
                "key_trends": ["EdTech", "Blended Learning", "Personalized Education", "Online Teaching"],
                "learning_path": "Teaching Certifications → Subject Expertise → Technology Skills",
                "career_progression": "Teacher → Senior Teacher → Coordinator → Principal"
            },
            "DESIGN": {
                "market_demand": "Cao",
                "salary_range": "15-70M+ VND",
                "growth_rate": "12-18% annually",
                "key_trends": ["UX/UI Design", "Sustainable Design", "Digital Transformation", "Brand Strategy"],
                "learning_path": "Design Courses → Software Mastery → Portfolio Building",
                "career_progression": "Junior Designer → Senior Designer → Art Director → Creative Director"
            }
        }
        
        insights = industry_insights.get(job_category.upper(), industry_insights["INFORMATION-TECHNOLOGY"])
        
        # Generate recommendations based on score
        if overall_score >= 80:
            recommendation_level = "Excellent"
            recommendation = "CV rất phù hợp với yêu cầu công việc. Có thể apply ngay."
        elif overall_score >= 60:
            recommendation_level = "Good"
            recommendation = "CV khá phù hợp. Cần cải thiện một số điểm nhỏ."
        elif overall_score >= 40:
            recommendation_level = "Fair"
            recommendation = "CV cần cải thiện đáng kể để phù hợp hơn."
        else:
            recommendation_level = "Poor"
            recommendation = "CV cần cải thiện nhiều để đáp ứng yêu cầu."
        
        return {
            "overall_assessment": {
                "score": overall_score,
                "level": recommendation_level,
                "recommendation": recommendation
            },
            "component_scores": {
                "ats_score": ats_score,
                "skills_match_score": skills_match_score,
                "quality_score": quality_score
            },
            "industry_insights": insights,
            "market_analysis": {
                "demand": insights["market_demand"],
                "salary_range": insights["salary_range"],
                "growth_rate": insights["growth_rate"],
                "key_trends": insights["key_trends"]
            },
            "career_guidance": {
                "learning_path": insights["learning_path"],
                "career_progression": insights["career_progression"],
                "next_steps": self._get_next_steps(overall_score, job_category)
            }
        }
    
    def _get_next_steps(self, score: float, job_category: str) -> List[str]:
        """Đưa ra next steps dựa trên score và ngành nghề"""
        
        if score >= 80:
            return [
                "Chuẩn bị cho interview",
                "Research về company culture",
                "Practice common interview questions",
                "Update portfolio với latest projects"
            ]
        elif score >= 60:
            return [
                "Học thêm missing skills quan trọng",
                "Cải thiện CV format và content",
                "Thực hành projects để build portfolio",
                "Network với professionals trong ngành"
            ]
        elif score >= 40:
            return [
                "Học fundamental skills của ngành",
                "Tham gia online courses và certifications",
                "Build practical projects",
                "Tìm mentor để hướng dẫn"
            ]
        else:
            return [
                "Học basic skills cần thiết",
                "Tham gia bootcamp hoặc training program",
                "Build foundation knowledge",
                "Tìm entry-level positions để gain experience"
            ]

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