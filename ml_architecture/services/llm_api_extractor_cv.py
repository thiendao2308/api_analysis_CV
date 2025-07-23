import openai
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env nếu có
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Đọc từ biến môi trường

def extract_cv_info_from_text(cv_text):
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
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # hoặc "gpt-4" nếu bạn có quyền
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )
    content = response.choices[0].message['content'].strip()
    return content 