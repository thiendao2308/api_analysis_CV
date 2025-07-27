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