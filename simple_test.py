#!/usr/bin/env python3
"""
Simple test script để debug OpenAI import
"""
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import openai
    print(f"OpenAI version: {openai.VERSION}")
    print("✅ OpenAI imported successfully")
    
    # Test OpenAI class
    print("Testing OpenAI class...")
    client = openai.OpenAI()
    print("✅ OpenAI client created successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 