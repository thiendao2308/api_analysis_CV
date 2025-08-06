import openai
import os
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load biến môi trường từ file .env nếu có
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Đọc từ biến môi trường

def extract_skills_from_jd(jd_text):
    logger.info(f"Starting LLM JD skills extraction for text length: {len(jd_text)}")
    
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not found, skipping LLM JD extraction")
        return ""
    
    try:
        prompt = f"""
Bạn là chuyên gia trích xuất skills từ Job Description cho nhiều ngành nghề. Hãy phân tích JD và trích xuất tất cả skills yêu cầu.

JOB DESCRIPTION:
{jd_text}

HƯỚNG DẪN TRÍCH XUẤT THEO NGÀNH NGHỀ:

1. CÔNG NGHỆ THÔNG TIN:
   - Technical requirements: Programming languages, frameworks, databases
   - Tools & platforms: Git, Docker, AWS, Azure, cloud services
   - Experience levels: Junior, Mid-level, Senior, Lead
   - Certifications: AWS, Azure, Google Cloud, etc.
   - Soft skills: Problem-solving, teamwork, communication

2. MARKETING & DIGITAL:
   - Digital platforms: Facebook, Instagram, Google Ads, LinkedIn
   - Tools & software: Canva, Photoshop, Google Analytics, HubSpot
   - Campaign management: SEO, content marketing, social media
   - Analytics & metrics: ROI tracking, performance analysis
   - Soft skills: Creativity, communication, data analysis

3. TÀI CHÍNH & KẾ TOÁN:
   - Software proficiency: Excel, QuickBooks, SAP, Oracle
   - Financial skills: Modeling, analysis, risk assessment
   - Regulatory knowledge: GAAP, IFRS, tax compliance
   - Certifications: CFA, CPA, ACCA, etc.
   - Soft skills: Attention to detail, analytical thinking

4. NHÂN SỰ & TUYỂN DỤNG:
   - HR systems: HRIS, ATS, payroll software
   - Processes: Recruitment, onboarding, performance management
   - Compliance: Labor laws, HR policies, regulations
   - Soft skills: Communication, empathy, conflict resolution
   - Experience: Employee relations, training, development

5. THIẾT KẾ & SÁNG TẠO:
   - Design tools: Photoshop, Illustrator, Figma, Sketch
   - Skills: UI/UX design, brand identity, visual communication
   - Portfolio requirements: Design projects, creative process
   - Knowledge: Design principles, color theory, typography
   - Soft skills: Creativity, attention to detail, client communication

6. BÁN HÀNG & KINH DOANH:
   - CRM systems: Salesforce, HubSpot, customer management
   - Sales skills: Negotiation, lead generation, relationship building
   - Business knowledge: Market analysis, business development
   - Metrics: Sales targets, conversion rates, revenue growth
   - Soft skills: Communication, persistence, relationship building

7. Y TẾ & CHĂM SÓC SỨC KHỎE:
   - Medical systems: EMR, patient management, medical software
   - Clinical skills: Patient care, procedures, documentation
   - Regulatory knowledge: Healthcare laws, medical compliance
   - Certifications: Medical licenses, specialized training
   - Soft skills: Empathy, attention to detail, stress management

8. GIÁO DỤC & ĐÀO TẠO:
   - Educational platforms: LMS, online teaching tools
   - Teaching skills: Curriculum development, student assessment
   - Educational technology: Digital tools, online learning
   - Certifications: Teaching licenses, educational technology
   - Soft skills: Communication, patience, adaptability

NGUYÊN TẮC TRÍCH XUẤT:
- Trích xuất cả technical và soft skills yêu cầu
- Bao gồm tools, platforms, certifications cần thiết
- Chú ý đến experience levels và seniority
- Loại bỏ requirements quá chung chung
- Ưu tiên skills có thể đo lường và đánh giá được

TRẢ VỀ: Danh sách skills được phân tách bằng dấu phẩy, không có giải thích thêm.
"""
        
        logger.info("Making OpenAI API call for JD skills...")
        
        # Use new OpenAI API format for openai>=1.0.0
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.2,
        )
        
        skills = response.choices[0].message.content.strip()
        logger.info(f"✅ LLM JD skills extraction successful, extracted skills: {skills}")
        return skills
        
    except Exception as e:
        logger.error(f"❌ LLM JD skills extraction failed: {e}")
        return ""

    