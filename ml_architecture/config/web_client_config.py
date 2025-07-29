import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class WebClientConfig:
    """Cấu hình cho web client integration"""
    
    # Web client URL
    WEB_CLIENT_URL = os.getenv("WEB_CLIENT_URL", "https://khoaluanai.vercel.app")
    
    # API Key cho web client
    WEB_CLIENT_API_KEY = os.getenv("WEB_CLIENT_API_KEY", "")
    
    # Mode: 'mock' hoặc 'real'
    INTEGRATION_MODE = os.getenv("INTEGRATION_MODE", "mock")
    
    # Timeout settings
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "1"))
    
    # Cache settings
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "web_client_url": cls.WEB_CLIENT_URL,
            "api_key": cls.WEB_CLIENT_API_KEY,
            "integration_mode": cls.INTEGRATION_MODE,
            "request_timeout": cls.REQUEST_TIMEOUT,
            "max_retries": cls.MAX_RETRIES,
            "retry_delay": cls.RETRY_DELAY,
            "cache_enabled": cls.CACHE_ENABLED,
            "cache_ttl": cls.CACHE_TTL
        }
    
    @classmethod
    def is_mock_mode(cls) -> bool:
        """Check if running in mock mode"""
        return cls.INTEGRATION_MODE.lower() == "mock"
    
    @classmethod
    def is_real_mode(cls) -> bool:
        """Check if running in real mode"""
        return cls.INTEGRATION_MODE.lower() == "real"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration"""
        if cls.is_real_mode():
            if not cls.WEB_CLIENT_API_KEY:
                print("⚠️  Warning: WEB_CLIENT_API_KEY not set for real mode")
                return False
            if not cls.WEB_CLIENT_URL:
                print("⚠️  Warning: WEB_CLIENT_URL not set for real mode")
                return False
        return True

# Job categories mapping
JOB_CATEGORIES = {
    "INFORMATION-TECHNOLOGY": {
        "name": "Công nghệ thông tin",
        "subcategories": [
            "SOFTWARE_DEVELOPER",
            "FULLSTACK_DEVELOPER", 
            "BACKEND_DEVELOPER",
            "FRONTEND_DEVELOPER",
            "MOBILE_DEVELOPER",
            "DEVOPS_ENGINEER",
            "DATA_SCIENTIST",
            "MACHINE_LEARNING_ENGINEER",
            "SYSTEM_ADMINISTRATOR",
            "QA_ENGINEER"
        ]
    },
    "MARKETING": {
        "name": "Marketing",
        "subcategories": [
            "DIGITAL_MARKETING",
            "CONTENT_MARKETING",
            "SOCIAL_MEDIA_MARKETING",
            "SEO_SPECIALIST",
            "MARKETING_MANAGER",
            "BRAND_MANAGER"
        ]
    },
    "FINANCE": {
        "name": "Tài chính",
        "subcategories": [
            "FINANCIAL_ANALYST",
            "ACCOUNTANT",
            "AUDITOR",
            "INVESTMENT_BANKER",
            "FINANCIAL_ADVISOR"
        ]
    },
    "HUMAN_RESOURCES": {
        "name": "Nhân sự",
        "subcategories": [
            "HR_MANAGER",
            "RECRUITER",
            "HR_SPECIALIST",
            "TALENT_ACQUISITION",
            "COMPENSATION_SPECIALIST"
        ]
    },
    "DESIGN": {
        "name": "Thiết kế",
        "subcategories": [
            "UI_UX_DESIGNER",
            "GRAPHIC_DESIGNER",
            "WEB_DESIGNER",
            "PRODUCT_DESIGNER",
            "VISUAL_DESIGNER"
        ]
    }
}

# Experience levels
EXPERIENCE_LEVELS = {
    "ENTRY_LEVEL": {
        "name": "Mới tốt nghiệp",
        "years": "0-2 năm"
    },
    "MID_LEVEL": {
        "name": "Có kinh nghiệm",
        "years": "2-5 năm"
    },
    "SENIOR_LEVEL": {
        "name": "Cấp cao",
        "years": "5+ năm"
    }
}

# API endpoints mapping
API_ENDPOINTS = {
    "get_user_cv": "/api/cv/{user_id}/{cv_id}",
    "get_user_cvs": "/api/user/{user_id}/cvs",
    "get_job_posting": "/api/jobs/{job_id}",
    "get_jobs_by_category": "/api/jobs/category/{category}",
    "save_analysis": "/api/analysis",
    "get_user_profile": "/api/user/{user_id}/profile",
    "update_user_preferences": "/api/user/{user_id}/preferences"
}