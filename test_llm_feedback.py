#!/usr/bin/env python3
"""
Test script for LLM Feedback Generator
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_llm_feedback_generator():
    """Test LLM Feedback Generator"""
    print("🧪 Testing LLM Feedback Generator")
    print("=" * 50)
    
    try:
        from ml_architecture.services.llm_feedback_generator import LLMFeedbackGenerator
        
        # Initialize generator
        generator = LLMFeedbackGenerator()
        print("✅ LLM Feedback Generator initialized")
        
        # Test data
        cv_analysis = {
            "skills": ["Python", "Django", "React", "SQL", "Git"],
            "experience": ["Full-stack Developer at TechCorp (2022-2024)"],
            "education": ["Bachelor in Computer Science"]
        }
        
        jd_analysis = {
            "extracted_skills": ["Python", "Django", "React", "Docker", "AWS", "CI/CD"]
        }
        
        matching_analysis = {
            "matching_skills": ["Python", "Django", "React"],
            "missing_skills": ["Docker", "AWS", "CI/CD"],
            "skills_match_score": 50.0
        }
        
        quality_analysis = {
            "quality_score": 0.75,
            "strengths": ["Cấu trúc CV rõ ràng", "Kỹ năng phù hợp"],
            "weaknesses": ["Thiếu kinh nghiệm cloud"]
        }
        
        overall_score = 65.0
        job_category = "INFORMATION-TECHNOLOGY"
        job_position = "FULLSTACK_DEVELOPER"
        
        # Generate intelligent feedback
        print("\n🔄 Generating intelligent feedback...")
        feedback_result = generator.generate_intelligent_feedback(
            cv_analysis=cv_analysis,
            jd_analysis=jd_analysis,
            matching_analysis=matching_analysis,
            quality_analysis=quality_analysis,
            overall_score=overall_score,
            job_category=job_category,
            job_position=job_position
        )
        
        print("\n📊 LLM Feedback Result:")
        print("-" * 30)
        
        if feedback_result:
            print(f"✅ Overall Assessment: {feedback_result.get('overall_assessment', 'N/A')}")
            print(f"✅ Strengths: {feedback_result.get('strengths', [])}")
            print(f"✅ Weaknesses: {feedback_result.get('weaknesses', [])}")
            print(f"✅ Specific Suggestions: {feedback_result.get('specific_suggestions', [])}")
            print(f"✅ Priority Actions: {feedback_result.get('priority_actions', [])}")
            print(f"✅ Encouragement: {feedback_result.get('encouragement', 'N/A')}")
        else:
            print("❌ No feedback generated")
        
        # Test quick feedback
        print("\n🔄 Testing quick feedback...")
        quick_feedback = generator.generate_quick_feedback(overall_score, job_category)
        print(f"✅ Quick Feedback: {quick_feedback}")
        
        print("\n🎉 LLM Feedback Generator test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing LLM Feedback Generator: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_llm_feedback_generator()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 