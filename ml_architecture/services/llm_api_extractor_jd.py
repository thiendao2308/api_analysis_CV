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
        prompt = (
            "Extract a list of skills from the following job description. "
            "Return only a comma-separated list of skills, no explanation.\n\n"
            f"Job Description:\n{jd_text}\n\nSkills:"
        )
        
        logger.info("Making OpenAI API call for JD skills...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # hoặc "gpt-4" nếu bạn có quyền
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.2,
            api_key=OPENAI_API_KEY,
        )
        
        skills = response.choices[0].message['content'].strip()
        logger.info(f"✅ LLM JD skills extraction successful, extracted skills: {skills}")
        return skills
        
    except Exception as e:
        logger.error(f"❌ LLM JD skills extraction failed: {e}")
        return ""

    