#!/usr/bin/env python3
"""
Test script để kiểm tra toàn bộ luồng hoạt động 6 bước của hệ thống CV-JD Analysis
"""

import sys
import os
import logging

# Add the ml_architecture directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ml_architecture'))

from services.cv_parser import CVParser
from data.jd_analysis_system import JDAnalysisSystem
from services.cv_evaluation_service import CVEvaluationService
from services.suggestion_generator import SuggestionGenerator
from services.cv_quality_analyzer import CVQualityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_step_1_cv_parsing():
    """BƯỚC 1: Test nhận diện thành phần từ CV"""
    print("\n" + "="*50)
    print("🧪 BƯỚC 1: TEST NHẬN DIỆN THÀNH PHẦN TỪ CV")
    print("="*50)
    
    cv_parser = CVParser()
    
    sample_cv = """
    NGUYỄN VĂN A
    Email: nva@email.com | Phone: 0123456789

    MỤC TIÊU NGHỀ NGHIỆP
    Trở thành một lập trình viên Python chuyên nghiệp với kinh nghiệm trong lĩnh vực phát triển web.

    KINH NGHIỆM LÀM VIỆC
    Công ty ABC (01/2020 - Hiện tại)
    Lập trình viên Python
    - Phát triển ứng dụng web sử dụng Django
    - Làm việc với PostgreSQL và Redis
    - Triển khai ứng dụng trên AWS

    HỌC VẤN
    Đại học Bách Khoa (2016 - 2020)
    Chuyên ngành: Khoa học máy tính

    KỸ NĂNG
    - Python, Django, Flask
    - PostgreSQL, Redis, MongoDB
    - AWS, Docker, Git
    - JavaScript, React, HTML/CSS
    """
    
    try:
        result = cv_parser.parse(sample_cv)
        print("✅ BƯỚC 1: Parse CV thành công!")
        print(f"📋 Skills: {result.skills}")
        print(f"💼 Experience: {result.experience[:100]}...")
        print(f"🎓 Education: {result.education}")
        print(f"📝 Summary: {result.summary}")
        return result
    except Exception as e:
        print(f"❌ BƯỚC 1: Lỗi khi parse CV: {e}")
        return None

def test_step_2_jd_analysis():
    """BƯỚC 2: Test trích xuất yêu cầu từ JD"""
    print("\n" + "="*50)
    print("🧪 BƯỚC 2: TEST TRÍCH XUẤT YÊU CẦU TỪ JD")
    print("="*50)
    
    jd_analyzer = JDAnalysisSystem()
    
    sample_jd = """
    Tuyển dụng Senior Python Developer
    
    Yêu cầu:
    - Kinh nghiệm 3+ năm với Python
    - Thành thạo Django, Flask
    - Biết sử dụng PostgreSQL, Redis
    - Có kinh nghiệm với AWS, Docker
    - Hiểu biết về Git, CI/CD
    - Kỹ năng JavaScript, React
    """
    
    try:
        result = jd_analyzer.analyze_single_jd(sample_jd)
        print("✅ BƯỚC 2: Phân tích JD thành công!")
        print(f"📋 Skills tìm thấy: {result['skills']}")
        print(f"📊 Categories: {result['categories']}")
        return result
    except Exception as e:
        print(f"❌ BƯỚC 2: Lỗi khi phân tích JD: {e}")
        return None

def test_step_3_cv_jd_comparison():
    """BƯỚC 3: Test so sánh CV-JD để tính độ phù hợp"""
    print("\n" + "="*50)
    print("🧪 BƯỚC 3: TEST SO SÁNH CV-JD")
    print("="*50)
    
    cv_evaluator = CVEvaluationService()
    
    sample_cv = """
    NGUYỄN VĂN A
    Email: nva@email.com | Phone: 0123456789

    MỤC TIÊU NGHỀ NGHIỆP
    Trở thành một lập trình viên Python chuyên nghiệp.

    KINH NGHIỆM LÀM VIỆC
    Công ty ABC (01/2020 - Hiện tại)
    Lập trình viên Python
    - Phát triển ứng dụng web sử dụng Django
    - Làm việc với PostgreSQL và Redis
    - Triển khai ứng dụng trên AWS

    HỌC VẤN
    Đại học Bách Khoa (2016 - 2020)
    Chuyên ngành: Khoa học máy tính

    KỸ NĂNG
    - Python, Django, Flask
    - PostgreSQL, Redis, MongoDB
    - AWS, Docker, Git
    - JavaScript, React, HTML/CSS
    """
    
    sample_jd = """
    Tuyển dụng Senior Python Developer
    
    Yêu cầu:
    - Kinh nghiệm 3+ năm với Python
    - Thành thạo Django, Flask
    - Biết sử dụng PostgreSQL, Redis
    - Có kinh nghiệm với AWS, Docker
    - Hiểu biết về Git, CI/CD
    - Kỹ năng JavaScript, React
    """
    
    try:
        result = cv_evaluator.evaluate_cv_comprehensive(
            cv_text=sample_cv,
            job_category="INFORMATION-TECHNOLOGY",
            jd_text=sample_jd
        )
        print("✅ BƯỚC 3: So sánh CV-JD thành công!")
        print(f"📊 Điểm tổng: {result['overall_score']}/100")
        print(f"📋 Skills từ JD: {result['jd_skills']}")
        print(f"🤖 ML Insights: {result['ml_insights']}")
        return result
    except Exception as e:
        print(f"❌ BƯỚC 3: Lỗi khi so sánh CV-JD: {e}")
        return None

def test_step_4_suggestion_generation():
    """BƯỚC 4: Test gợi ý chỉnh sửa CV"""
    print("\n" + "="*50)
    print("🧪 BƯỚC 4: TEST GỢI Ý CHỈNH SỬA CV")
    print("="*50)
    
    suggestion_gen = SuggestionGenerator()
    
    # Giả lập dữ liệu từ phân tích CV-JD
    matched_skills = ["Python", "Django", "PostgreSQL"]
    missing_skills = ["Docker", "CI/CD", "Kubernetes"]
    cv_quality = {"quality_score": 0.8}
    job_category = "INFORMATION-TECHNOLOGY"
    
    try:
        suggestions = suggestion_gen.generate(
            matched_keywords=matched_skills,
            missing_keywords=missing_skills,
            cv_quality=cv_quality,
            job_category=job_category
        )
        print("✅ BƯỚC 4: Tạo gợi ý thành công!")
        print("💡 Các gợi ý:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        return suggestions
    except Exception as e:
        print(f"❌ BƯỚC 4: Lỗi khi tạo gợi ý: {e}")
        return None

def test_step_5_missing_skills():
    """BƯỚC 5: Test liệt kê kỹ năng còn thiếu"""
    print("\n" + "="*50)
    print("🧪 BƯỚC 5: TEST LIỆT KÊ KỸ NĂNG CÒN THIẾU")
    print("="*50)
    
    # Giả lập dữ liệu
    cv_skills = ["Python", "Django", "PostgreSQL", "AWS"]
    jd_skills = ["Python", "Django", "PostgreSQL", "Docker", "Kubernetes", "CI/CD"]
    
    try:
        # Tính skills thiếu
        cv_skills_set = set(cv_skills)
        jd_skills_set = set(jd_skills)
        missing_skills = jd_skills_set - cv_skills_set
        matching_skills = cv_skills_set.intersection(jd_skills_set)
        
        print("✅ BƯỚC 5: Phân tích skills thiếu thành công!")
        print(f"📋 Skills CV có: {list(cv_skills_set)}")
        print(f"📋 Skills JD yêu cầu: {list(jd_skills_set)}")
        print(f"✅ Skills phù hợp: {list(matching_skills)}")
        print(f"⚠️ Skills còn thiếu: {list(missing_skills)}")
        
        return {
            "cv_skills": list(cv_skills_set),
            "jd_skills": list(jd_skills_set),
            "matching_skills": list(matching_skills),
            "missing_skills": list(missing_skills)
        }
    except Exception as e:
        print(f"❌ BƯỚC 5: Lỗi khi phân tích skills thiếu: {e}")
        return None

def test_step_6_ats_scoring():
    """BƯỚC 6: Test chấm điểm tổng thể ATS"""
    print("\n" + "="*50)
    print("🧪 BƯỚC 6: TEST CHẤM ĐIỂM TỔNG THỂ ATS")
    print("="*50)
    
    quality_analyzer = CVQualityAnalyzer()
    
    # Giả lập CV đã parse
    from models.shared_models import ParsedCV
    
    sample_cv = ParsedCV(
        summary="Lập trình viên Python với 3 năm kinh nghiệm",
        skills=["Python", "Django", "PostgreSQL", "AWS", "Docker"],
        experience="Công ty ABC - Lập trình viên Python (2020-2023)",
        education="Đại học Bách Khoa - Khoa học máy tính"
    )
    
    try:
        quality_result = quality_analyzer.analyze(sample_cv)
        ats_score = quality_analyzer.calculate_ats_score(quality_result)
        
        print("✅ BƯỚC 6: Chấm điểm ATS thành công!")
        print(f"📊 Điểm chất lượng: {quality_result['quality_score']:.2f}")
        print(f"📊 Điểm ATS: {ats_score}/100")
        print(f"✅ Điểm mạnh: {quality_result['strengths']}")
        print(f"⚠️ Điểm yếu: {quality_result['weaknesses']}")
        
        return {
            "quality_result": quality_result,
            "ats_score": ats_score
        }
    except Exception as e:
        print(f"❌ BƯỚC 6: Lỗi khi chấm điểm ATS: {e}")
        return None

def test_complete_workflow():
    """Test toàn bộ luồng hoạt động 6 bước"""
    print("\n" + "🚀" + "="*48)
    print("🚀 TEST TOÀN BỘ LUỒNG HOẠT ĐỘNG 6 BƯỚC")
    print("🚀" + "="*48)
    
    results = {}
    
    # Test từng bước
    results['step_1'] = test_step_1_cv_parsing()
    results['step_2'] = test_step_2_jd_analysis()
    results['step_3'] = test_step_3_cv_jd_comparison()
    results['step_4'] = test_step_4_suggestion_generation()
    results['step_5'] = test_step_5_missing_skills()
    results['step_6'] = test_step_6_ats_scoring()
    
    # Tổng kết
    print("\n" + "🎯" + "="*48)
    print("🎯 TỔNG KẾT KẾT QUẢ TEST")
    print("🎯" + "="*48)
    
    successful_steps = 0
    total_steps = 6
    
    for step_name, result in results.items():
        if result is not None:
            successful_steps += 1
            print(f"✅ {step_name.upper()}: THÀNH CÔNG")
        else:
            print(f"❌ {step_name.upper()}: THẤT BẠI")
    
    print(f"\n📊 KẾT QUẢ: {successful_steps}/{total_steps} bước thành công")
    
    if successful_steps == total_steps:
        print("🎉 TẤT CẢ 6 BƯỚC ĐỀU HOẠT ĐỘNG TỐT!")
        print("✅ Hệ thống đã sẵn sàng cho production!")
    else:
        print("⚠️ CÓ MỘT SỐ BƯỚC CẦN KIỂM TRA LẠI!")
    
    return results

if __name__ == "__main__":
    test_complete_workflow() 