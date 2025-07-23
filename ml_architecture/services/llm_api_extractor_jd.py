import openai
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env nếu có
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Đọc từ biến môi trường

def extract_skills_from_jd(jd_text):
    prompt = (
        "Extract a list of skills from the following job description. "
        "Return only a comma-separated list of skills, no explanation.\n\n"
        f"Job Description:\n{jd_text}\n\nSkills:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # hoặc "gpt-4" nếu bạn có quyền
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )
    skills = response.choices[0].message['content'].strip()
    return skills

    