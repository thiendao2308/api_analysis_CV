#!/usr/bin/env python3
"""
Test script để kiểm tra OpenAI API
"""
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def test_openai_api():
    """Test OpenAI API với format mới"""
    if not OPENAI_API_KEY:
        print("❌ OpenAI API key not found")
        return False
    
    try:
        print("🔧 Testing OpenAI API...")
        
        # Use new OpenAI API format
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, test message"}],
            max_tokens=50,
            temperature=0.2,
        )
        
        content = response.choices[0].message.content
        print(f"✅ OpenAI API test successful: {content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

if __name__ == "__main__":
    test_openai_api() 