#!/usr/bin/env python3
"""
Simple test script for CV Analysis API
"""
import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_user_cv():
    """Test get user CV endpoint"""
    print("\n🔍 Testing get user CV...")
    try:
        response = requests.get(f"{BASE_URL}/get-user-cv/user_1/cv_1")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_get_job_posting():
    """Test get job posting endpoint"""
    print("\n🔍 Testing get job posting...")
    try:
        response = requests.get(f"{BASE_URL}/get-job-posting/job_1")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze_cv_from_web_client():
    """Test analyze CV from web client endpoint"""
    print("\n🔍 Testing analyze CV from web client...")
    try:
        payload = {
            "user_id": "user_1",
            "cv_id": "cv_1",
            "job_id": "job_1",
            "job_category": "INFORMATION-TECHNOLOGY",
            "job_position": "FULLSTACK_DEVELOPER"
        }
        
        response = requests.post(
            f"{BASE_URL}/analyze-cv-from-web-client",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis completed successfully!")
            print(f"Overall Score: {result.get('scores', {}).get('overall_score', 'N/A')}")
            print(f"Match Score: {result.get('matching_analysis', {}).get('skills_match_score', 'N/A')}%")
        else:
            print(f"❌ Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_analyze_cv_with_job():
    """Test analyze CV with job suggestions endpoint"""
    print("\n🔍 Testing analyze CV with job suggestions...")
    try:
        payload = {
            "user_id": "user_1",
            "cv_id": "cv_1",
            "job_category": "INFORMATION-TECHNOLOGY",
            "job_position": "FULLSTACK_DEVELOPER"
        }
        
        response = requests.post(
            f"{BASE_URL}/analyze-cv-with-job",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis completed successfully!")
            print(f"Number of suggestions: {len(result.get('suggested_analyses', []))}")
        else:
            print(f"❌ Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API tests...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Get User CV", test_get_user_cv),
        ("Get Job Posting", test_get_job_posting),
        ("Analyze CV from Web Client", test_analyze_cv_from_web_client),
        ("Analyze CV with Job Suggestions", test_analyze_cv_with_job)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()