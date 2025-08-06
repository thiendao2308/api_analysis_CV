import openai
import os
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load biến môi trường từ file .env nếu có
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Đọc từ biến môi trường

def extract_cv_info_from_text(cv_text):
    logger.info(f"Starting LLM CV extraction for text length: {len(cv_text)}")
    
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not found, skipping LLM extraction")
        return "{}"
    
    try:
        prompt = (
            "Extract the following information from the CV text below (if available):\n"
            "- Work experience: position, company, duration\n"
            "- Education: degree, school, year\n"
            "- Projects: name, description, role\n"
            "- Certifications: name, organization, year\n"
            "- Skills\n\n"
            "Return the result as valid JSON.\n\n"
            f"CV Text:\n\"\"\"\n{cv_text}\n\"\"\"\n"
        )
        
        logger.info("Making OpenAI API call...")
        
        # Use new OpenAI API format for openai>=1.0.0
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.2,
        )
        
        content = response.choices[0].message.content.strip()
        logger.info(f"✅ LLM CV extraction successful, response length: {len(content)}")
        return content
        
    except Exception as e:
        logger.error(f"❌ LLM CV extraction failed: {e}")
        return "{}"

def extract_skills_from_cv(cv_text):
    logger.info(f"Starting LLM CV skills extraction for text length: {len(cv_text)}")
    
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not found, skipping LLM CV extraction")
        return ""
    
    try:
        prompt = f"""
Bạn là chuyên gia trích xuất skills từ CV cho nhiều ngành nghề. Hãy phân tích CV và trích xuất tất cả skills liên quan.

CV TEXT:
{cv_text}

HƯỚNG DẪN TRÍCH XUẤT THEO NGÀNH NGHỀ:

1. CÔNG NGHỆ THÔNG TIN:
   - Technical skills: Programming languages, frameworks, databases, cloud platforms
   - Tools: Git, Docker, Jenkins, AWS, Azure, etc.
   - Soft skills: Problem-solving, analytical thinking, teamwork
   - Certifications: AWS, Azure, Google Cloud, etc.

2. MARKETING & DIGITAL:
   - Digital platforms: Facebook Ads, Google Ads, Instagram, LinkedIn
   - Tools: Canva, Photoshop, Google Analytics, HubSpot
   - Skills: SEO, content creation, social media management
   - Analytics: Data analysis, campaign performance, ROI tracking

3. TÀI CHÍNH & KẾ TOÁN:
   - Software: Excel, QuickBooks, SAP, Oracle
   - Skills: Financial modeling, risk analysis, compliance
   - Certifications: CFA, CPA, ACCA, etc.
   - Knowledge: GAAP, IFRS, tax regulations

4. NHÂN SỰ & TUYỂN DỤNG:
   - Systems: HRIS, ATS, payroll systems
   - Processes: Recruitment, onboarding, performance management
   - Skills: Employee relations, conflict resolution, training
   - Knowledge: Labor laws, HR policies, compliance

5. THIẾT KẾ & SÁNG TẠO:
   - Tools: Photoshop, Illustrator, Figma, Sketch
   - Skills: UI/UX design, brand identity, visual communication
   - Portfolio: Design projects, creative process
   - Knowledge: Design principles, color theory, typography

6. BÁN HÀNG & KINH DOANH:
   - Systems: CRM, Salesforce, HubSpot
   - Skills: Negotiation, lead generation, customer relationship
   - Knowledge: Sales techniques, market analysis, business development
   - Metrics: Sales targets, conversion rates, revenue growth

7. Y TẾ & CHĂM SÓC SỨC KHỎE:
   - Systems: EMR, patient management, medical software
   - Skills: Patient care, clinical procedures, medical documentation
   - Knowledge: Medical terminology, healthcare regulations
   - Certifications: Medical licenses, specialized training

8. GIÁO DỤC & ĐÀO TẠO:
   - Platforms: LMS, online teaching tools, educational software
   - Skills: Curriculum development, student assessment, teaching methods
   - Knowledge: Pedagogy, educational psychology, learning theories
   - Certifications: Teaching licenses, educational technology

NGUYÊN TẮC TRÍCH XUẤT:
- Trích xuất cả technical và soft skills
- Bao gồm tools, platforms, certifications
- Chú ý đến industry-specific terminology
- Loại bỏ skills quá chung chung hoặc không liên quan
- Ưu tiên skills có thể đo lường được

TRẢ VỀ: Danh sách skills được phân tách bằng dấu phẩy, không có giải thích thêm.
"""
        
        logger.info("Making OpenAI API call...")
        
        # Use new OpenAI API format for openai>=1.0.0
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.2,
        )
        
        skills = response.choices[0].message.content.strip()
        logger.info(f"✅ LLM CV skills extraction successful, response length: {len(skills)}")
        return skills
        
    except Exception as e:
        logger.error(f"❌ LLM CV skills extraction failed: {e}")
        return "" 