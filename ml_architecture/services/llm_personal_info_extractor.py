import logging
import openai
import os
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PersonalInfo:
    """Thông tin cá nhân trích xuất từ CV - chỉ cần thiết"""
    full_name: str
    job_position: Optional[str] = None

class LLMPersonalInfoExtractor:
    """Trích xuất thông tin cá nhân từ CV sử dụng LLM (OpenAI)"""
    
    def __init__(self):
        self.client = None
        self._init_openai_client()
    
    def _init_openai_client(self):
        """Khởi tạo OpenAI client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("✅ OpenAI client initialized successfully")
            else:
                logger.warning("⚠️ OPENAI_API_KEY not found in environment variables")
        except Exception as e:
            logger.error(f"❌ Error initializing OpenAI client: {e}")
    
    def extract_personal_info(self, cv_text: str) -> PersonalInfo:
        """Trích xuất thông tin cá nhân từ CV sử dụng LLM"""
        try:
            if not self.client:
                logger.warning("⚠️ OpenAI client not available, using fallback")
                return self._fallback_extraction(cv_text)
            
            # Tạo prompt cho LLM
            prompt = self._create_extraction_prompt(cv_text)
            
            # Gọi OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một AI assistant chuyên trích xuất thông tin cá nhân từ CV. Hãy trích xuất chính xác tên ứng viên và vị trí công việc họ đang ứng tuyển."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Thấp để có kết quả nhất quán
                max_tokens=200
            )
            
            # Xử lý response từ LLM
            llm_response = response.choices[0].message.content
            logger.info(f"🤖 LLM Response: {llm_response}")
            
            # Parse kết quả từ LLM
            personal_info = self._parse_llm_response(llm_response)
            
            if personal_info.full_name != "Ứng viên":
                logger.info(f"✅ LLM extracted: {personal_info.full_name} applying for {personal_info.job_position}")
            else:
                logger.warning("⚠️ LLM extraction failed, using fallback")
                personal_info = self._fallback_extraction(cv_text)
            
            return personal_info
            
        except Exception as e:
            logger.error(f"❌ Error in LLM extraction: {e}")
            return self._fallback_extraction(cv_text)
    
    def _create_extraction_prompt(self, cv_text: str) -> str:
        """Tạo prompt cho LLM"""
        # Lấy 20 dòng đầu của CV để giảm token usage
        lines = cv_text.split('\n')
        cv_preview = '\n'.join(lines[:20])
        
        prompt = f"""
Hãy trích xuất thông tin cá nhân từ CV sau đây:

CV TEXT:
{cv_preview}

Yêu cầu:
1. Tìm tên đầy đủ của ứng viên (thường ở dòng đầu tiên)
2. Tìm vị trí công việc họ đang ứng tuyển (backend developer, frontend developer, etc.)

Hãy trả về kết quả theo format JSON:
{{
    "full_name": "Tên đầy đủ của ứng viên",
    "job_position": "Vị trí công việc"
}}

Lưu ý:
- Tên phải chính xác, không được thay đổi
- Vị trí công việc phải rõ ràng (backend, frontend, fullstack, developer, engineer, etc.)
- Nếu không tìm thấy thông tin, hãy để null
"""
        return prompt
    
    def _parse_llm_response(self, llm_response: str) -> PersonalInfo:
        """Parse response từ LLM"""
        try:
            # Tìm JSON trong response
            import json
            import re
            
            # Tìm JSON pattern
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                full_name = data.get('full_name', 'Ứng viên')
                job_position = data.get('job_position')
                
                # Validate tên
                if full_name and full_name != "Ứng viên" and len(full_name.strip()) > 2:
                    return PersonalInfo(
                        full_name=full_name.strip(),
                        job_position=job_position.strip() if job_position else None
                    )
            
            # Fallback: tìm tên và vị trí trong text response
            return self._extract_from_text_response(llm_response)
            
        except Exception as e:
            logger.error(f"❌ Error parsing LLM response: {e}")
            return self._extract_from_text_response(llm_response)
    
    def _extract_from_text_response(self, llm_response: str) -> PersonalInfo:
        """Trích xuất thông tin từ text response nếu JSON parse thất bại"""
        try:
            lines = llm_response.split('\n')
            full_name = "Ứng viên"
            job_position = None
            
            for line in lines:
                line_lower = line.lower()
                
                # Tìm tên
                if 'tên' in line_lower or 'name' in line_lower:
                    if ':' in line:
                        name_part = line.split(':')[1].strip()
                        if len(name_part) > 2:
                            full_name = name_part
                
                # Tìm vị trí
                if 'vị trí' in line_lower or 'position' in line_lower or 'job' in line_lower:
                    if ':' in line:
                        position_part = line.split(':')[1].strip()
                        if len(position_part) > 2:
                            job_position = position_part
            
            return PersonalInfo(full_name=full_name, job_position=job_position)
            
        except Exception as e:
            logger.error(f"❌ Error extracting from text response: {e}")
            return PersonalInfo(full_name="Ứng viên")
    
    def _fallback_extraction(self, cv_text: str) -> PersonalInfo:
        """Fallback extraction khi LLM không khả dụng"""
        try:
            lines = cv_text.split('\n')
            
            # Tìm tên ở dòng đầu
            full_name = "Ứng viên"
            if lines:
                first_line = lines[0].strip()
                if len(first_line) > 3 and len(first_line) < 100:
                    # Kiểm tra xem có phải là tên tiếng Việt không
                    if any(char in first_line for char in 'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ'):
                        if not any(word.lower() in ['cv', 'resume', 'curriculum vitae', 'sơ yếu lý lịch', 'backend', 'frontend', 'developer', 'engineer'] 
                                 for word in first_line.split()):
                            full_name = first_line
            
            # Tìm vị trí công việc
            job_position = None
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['backend', 'frontend', 'fullstack', 'developer', 'engineer', 'programmer']):
                    for keyword in ['backend', 'frontend', 'fullstack', 'developer', 'engineer', 'programmer']:
                        if keyword in line_lower:
                            job_position = keyword.title()
                            break
                    if job_position:
                        break
            
            return PersonalInfo(full_name=full_name, job_position=job_position)
            
        except Exception as e:
            logger.error(f"❌ Error in fallback extraction: {e}")
            return PersonalInfo(full_name="Ứng viên")
