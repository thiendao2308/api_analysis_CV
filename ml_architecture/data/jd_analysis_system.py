import spacy
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import os

class JDAnalysisSystem:
    """
    BƯỚC 2: Trích xuất yêu cầu từ JD (NLP/NER model)
    Hệ thống phân tích JD và trích xuất skills requirements
    """
    def __init__(self, model_path="model_full/model-best"):
        """Khởi tạo hệ thống phân tích JD - BƯỚC 2"""
        try:
            # Sử dụng đường dẫn tuyệt đối
            abs_model_path = os.path.join(os.path.dirname(__file__), model_path)
            self.nlp = spacy.load(abs_model_path)
            print(f"✅ BƯỚC 2: Đã load model: {abs_model_path}")
        except OSError:
            try:
                # Fallback sang model cũ
                abs_model_path = os.path.join(os.path.dirname(__file__), "model", "model-best")
                self.nlp = spacy.load(abs_model_path)
                print(f"✅ BƯỚC 2: Đã load model fallback: {abs_model_path}")
            except OSError:
                print(f"❌ BƯỚC 2: Không thể load model từ {model_path}")
                print("Sử dụng model mặc định...")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except:
                    print("❌ BƯỚC 2: Không có model nào khả dụng!")
                    self.nlp = None
    
    def extract_skills_from_jd(self, jd_text):
        """BƯỚC 2: Trích xuất skills từ JD sử dụng NER model"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(jd_text)
            skills = []
            
            for ent in doc.ents:
                if ent.label_ == "SKILL":
                    skill_text = ent.text.strip()
                    if skill_text and len(skill_text) > 1:
                        skills.append(skill_text)
            
            return list(set(skills))  # Loại bỏ duplicates
        except Exception as e:
            print(f"❌ BƯỚC 2: Lỗi khi trích xuất skills từ JD: {e}")
            return []
    
    def analyze_single_jd(self, jd_text):
        """BƯỚC 2: Phân tích một JD đơn lẻ"""
        print("=== BƯỚC 2: PHÂN TÍCH JD ===")
        print(f"JD: {jd_text[:200]}...")
        
        # Trích xuất skills
        skills = self.extract_skills_from_jd(jd_text)
        
        print(f"\n📋 Skills tìm thấy ({len(skills)}):")
        for i, skill in enumerate(skills, 1):
            print(f"  {i}. {skill}")
        
        # Phân tích loại skills
        skill_categories = self.categorize_skills(skills)
        
        print(f"\n📊 Phân loại skills:")
        for category, skills_list in skill_categories.items():
            if skills_list:
                print(f"  {category}: {', '.join(skills_list)}")
        
        return {
            'skills': skills,
            'skill_count': len(skills),
            'categories': skill_categories
        }
    
    def categorize_skills(self, skills):
        """Phân loại skills thành các nhóm"""
        categories = {
            'Programming Languages': [],
            'Frameworks & Libraries': [],
            'Databases': [],
            'Cloud & DevOps': [],
            'Soft Skills': [],
            'Tools & Platforms': [],
            'Other': []
        }
        
        # Định nghĩa từ khóa cho từng category
        keywords = {
            'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala'],
            'Frameworks & Libraries': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'express', 'jquery', 'bootstrap', 'tensorflow', 'pytorch'],
            'Databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sql server', 'sqlite', 'elasticsearch'],
            'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform', 'ansible'],
            'Soft Skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'analytical', 'creativity', 'time management'],
            'Tools & Platforms': ['jira', 'confluence', 'slack', 'teams', 'figma', 'adobe', 'excel', 'powerpoint', 'word']
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            categorized = False
            
            for category, keyword_list in keywords.items():
                if any(keyword in skill_lower for keyword in keyword_list):
                    categories[category].append(skill)
                    categorized = True
                    break
            
            if not categorized:
                categories['Other'].append(skill)
        
        return categories
    
    def analyze_multiple_jds(self, jd_list):
        """BƯỚC 2: Phân tích nhiều JD"""
        print("=== BƯỚC 2: PHÂN TÍCH NHIỀU JD ===")
        
        all_skills = []
        all_categories = []
        
        for i, jd in enumerate(jd_list):
            print(f"\nJD {i+1}:")
            result = self.analyze_single_jd(jd)
            all_skills.extend(result['skills'])
            all_categories.append(result['categories'])
        
        # Thống kê tổng hợp
        skill_counts = Counter(all_skills)
        
        print(f"\n📈 THỐNG KÊ TỔNG HỢP:")
        print(f"Tổng số skills: {len(all_skills)}")
        print(f"Số skills unique: {len(skill_counts)}")
        
        print(f"\n🏆 TOP 10 SKILLS PHỔ BIẾN:")
        for skill, count in skill_counts.most_common(10):
            print(f"  {skill}: {count} lần")
        
        return {
            'all_skills': all_skills,
            'skill_counts': skill_counts,
            'all_categories': all_categories
        }
    
    def analyze_jd_file(self, file_path):
        """BƯỚC 2: Phân tích file CSV chứa JD"""
        print(f"=== BƯỚC 2: PHÂN TÍCH FILE: {file_path} ===")
        
        # Đọc file
        try:
            df = pd.read_csv(file_path)
            if 'text' not in df.columns:
                df.columns = ['text']
        except:
            df = pd.read_csv(file_path, header=None, names=['text'])
        
        print(f"Tổng số JD: {len(df)}")
        
        # Phân tích từng JD
        all_skills = []
        skill_counts = Counter()
        
        for idx, row in df.iterrows():
            jd_text = row['text']
            skills = self.extract_skills_from_jd(jd_text)
            all_skills.extend(skills)
            skill_counts.update(skills)
            
            if (idx + 1) % 100 == 0:
                print(f"Đã phân tích {idx + 1}/{len(df)} JD...")
        
        # Tạo báo cáo
        self.generate_report(skill_counts, len(df))
        
        return {
            'total_jds': len(df),
            'total_skills': len(all_skills),
            'unique_skills': len(skill_counts),
            'skill_counts': skill_counts
        }
    
    def generate_report(self, skill_counts, total_jds):
        """Tạo báo cáo phân tích"""
        print(f"\n📊 BÁO CÁO PHÂN TÍCH:")
        print(f"Tổng số JD: {total_jds:,}")
        print(f"Tổng số skills: {sum(skill_counts.values()):,}")
        print(f"Số skills unique: {len(skill_counts):,}")
        print(f"Trung bình skills/JD: {sum(skill_counts.values())/total_jds:.1f}")
        
        print(f"\n🏆 TOP 20 SKILLS PHỔ BIẾN:")
        for skill, count in skill_counts.most_common(20):
            percentage = (count / total_jds) * 100
            print(f"  {skill}: {count:,} lần ({percentage:.1f}%)")
    
    def find_similar_jds(self, target_jd, jd_list, top_k=5):
        """Tìm JD tương tự"""
        print("=== BƯỚC 2: TÌM JD TƯƠNG TỰ ===")
        
        # Trích xuất skills của target JD
        target_skills = set(self.extract_skills_from_jd(target_jd))
        
        similarities = []
        
        for i, jd in enumerate(jd_list):
            jd_skills = set(self.extract_skills_from_jd(jd))
            
            # Tính Jaccard similarity
            if target_skills or jd_skills:
                intersection = len(target_skills.intersection(jd_skills))
                union = len(target_skills.union(jd_skills))
                similarity = intersection / union if union > 0 else 0
                similarities.append((i, similarity, jd_skills))
        
        # Sắp xếp theo similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP {top_k} JD TƯƠNG TỰ:")
        for i, (jd_idx, similarity, skills) in enumerate(similarities[:top_k]):
            print(f"  {i+1}. JD {jd_idx+1}: {similarity:.2f} ({len(skills)} skills)")
        
        return similarities[:top_k]

# Test function
if __name__ == "__main__":
    # Test với JD mẫu
    jd_analyzer = JDAnalysisSystem()
    
    sample_jd = """
    We are looking for a Senior Python Developer with experience in:
    - Python programming
    - Django framework
    - PostgreSQL database
    - AWS cloud services
    - Docker containerization
    - Git version control
    """
    
    result = jd_analyzer.analyze_single_jd(sample_jd)
    print(f"\nKết quả phân tích:")
    print(f"Skills: {result['skills']}")
    print(f"Categories: {result['categories']}") 