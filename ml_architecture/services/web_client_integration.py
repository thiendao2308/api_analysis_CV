import aiohttp
import logging
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class WebClientIntegration:
    """Service để tích hợp với web client"""
    
    def __init__(self):
        self.web_client_url = os.getenv("WEB_CLIENT_URL", "https://khoaluanai.vercel.app")
        self.api_key = os.getenv("WEB_CLIENT_API_KEY", "")
        self.session = None
    
    async def get_session(self):
        """Lazy load aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_cv_content(self, user_id: str, cv_id: str) -> str:
        """
        Lấy CV content từ web client
        """
        try:
            session = await self.get_session()
            url = f"{self.web_client_url}/api/cv/{user_id}/{cv_id}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('content', '')
                else:
                    logger.error(f"Failed to get CV content: {response.status}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error getting CV content: {e}")
            return ""
    
    async def get_job_content(self, job_id: str) -> Dict:
        """
        Lấy job content từ web client
        """
        try:
            session = await self.get_session()
            url = f"{self.web_client_url}/api/jobs/{job_id}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get job content: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting job content: {e}")
            return {}
    
    async def get_user_cvs(self, user_id: str) -> List[Dict]:
        """
        Lấy danh sách CV của user
        """
        try:
            session = await self.get_session()
            url = f"{self.web_client_url}/api/user/{user_id}/cvs"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get user CVs: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting user CVs: {e}")
            return []
    
    async def get_jobs_by_category(self, category: str) -> List[Dict]:
        """
        Lấy danh sách jobs theo category
        """
        try:
            session = await self.get_session()
            url = f"{self.web_client_url}/api/jobs/category/{category}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get jobs by category: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting jobs by category: {e}")
            return []
    
    async def save_analysis_result(self, user_id: str, cv_id: str, job_id: str, analysis_result: Dict) -> bool:
        """
        Lưu kết quả phân tích vào web client
        """
        try:
            session = await self.get_session()
            url = f"{self.web_client_url}/api/analysis"
            
            payload = {
                "user_id": user_id,
                "cv_id": cv_id,
                "job_id": job_id,
                "analysis_result": analysis_result,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Analysis result saved successfully")
                    return True
                else:
                    logger.error(f"Failed to save analysis result: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Dict:
        """
        Lấy thông tin profile của user
        """
        try:
            session = await self.get_session()
            url = f"{self.web_client_url}/api/user/{user_id}/profile"
            
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get user profile: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {}
    
    async def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """
        Cập nhật preferences của user
        """
        try:
            session = await self.get_session()
            url = f"{self.web_client_url}/api/user/{user_id}/preferences"
            
            async with session.put(url, json=preferences) as response:
                if response.status == 200:
                    logger.info(f"User preferences updated successfully")
                    return True
                else:
                    logger.error(f"Failed to update user preferences: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False

# Mock implementation cho development
class MockWebClientIntegration:
    """Mock implementation cho development khi chưa có web client API"""
    
    def __init__(self):
        self.mock_cvs = {
            "user_1": {
                "cv_1": {
                    "content": """
                    NGUYỄN VĂN A
                    Software Engineer
                    
                    KINH NGHIỆM
                    - 3 năm kinh nghiệm phát triển web với Python, Django
                    - 2 năm kinh nghiệm với React, JavaScript
                    - Thành thạo Git, Docker, AWS
                    
                    KỸ NĂNG
                    - Python, Django, Flask
                    - JavaScript, React, Node.js
                    - SQL, PostgreSQL, MongoDB
                    - Git, Docker, AWS
                    """,
                    "filename": "cv_nguyen_van_a.pdf",
                    "upload_date": "2024-01-01"
                }
            }
        }
        
        self.mock_jobs = {
            "job_1": {
                "job_title": "Full Stack Developer",
                "company_name": "Tech Company A",
                "job_description": """
                Chúng tôi đang tìm kiếm một Full Stack Developer có kinh nghiệm với:
                - Python, Django, Flask
                - JavaScript, React, Node.js
                - Database: PostgreSQL, MongoDB
                - Cloud: AWS, Docker
                """,
                "job_requirements": """
                Yêu cầu:
                - 2+ năm kinh nghiệm
                - Thành thạo Python và JavaScript
                - Có kinh nghiệm với cloud platforms
                """
            },
            "job_2": {
                "job_title": "Backend Developer",
                "company_name": "Tech Company B", 
                "job_description": """
                Tuyển Backend Developer với kỹ năng:
                - Python, Django, FastAPI
                - Database design và optimization
                - API development và documentation
                - Microservices architecture
                """,
                "job_requirements": """
                Yêu cầu:
                - 3+ năm kinh nghiệm backend
                - Thành thạo Python frameworks
                - Có kinh nghiệm với microservices
                """
            }
        }
    
    async def get_cv_content(self, user_id: str, cv_id: str) -> str:
        """Mock get CV content"""
        user_cvs = self.mock_cvs.get(user_id, {})
        cv_data = user_cvs.get(cv_id, {})
        return cv_data.get('content', '')
    
    async def get_job_content(self, job_id: str) -> Dict:
        """Mock get job content"""
        return self.mock_jobs.get(job_id, {})
    
    async def get_user_cvs(self, user_id: str) -> List[Dict]:
        """Mock get user CVs"""
        user_cvs = self.mock_cvs.get(user_id, {})
        return [
            {
                "cv_id": cv_id,
                "filename": cv_data.get('filename', ''),
                "upload_date": cv_data.get('upload_date', '')
            }
            for cv_id, cv_data in user_cvs.items()
        ]
    
    async def get_jobs_by_category(self, category: str) -> List[Dict]:
        """Mock get jobs by category"""
        return [
            {
                "job_id": job_id,
                "job_title": job_data.get('job_title', ''),
                "company_name": job_data.get('company_name', '')
            }
            for job_id, job_data in self.mock_jobs.items()
        ]
    
    async def save_analysis_result(self, user_id: str, cv_id: str, job_id: str, analysis_result: Dict) -> bool:
        """Mock save analysis result"""
        logger.info(f"Mock: Saving analysis result for user {user_id}, CV {cv_id}, job {job_id}")
        return True
    
    async def get_user_profile(self, user_id: str) -> Dict:
        """Mock get user profile"""
        return {
            "user_id": user_id,
            "name": "Nguyễn Văn A",
            "email": "nguyenvana@email.com",
            "preferences": {
                "job_categories": ["INFORMATION-TECHNOLOGY"],
                "experience_level": "MID_LEVEL"
            }
        }
    
    async def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Mock update user preferences"""
        logger.info(f"Mock: Updating preferences for user {user_id}")
        return True