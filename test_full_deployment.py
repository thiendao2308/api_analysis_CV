#!/usr/bin/env python3
"""
Test script for full deployment with all dependencies
"""
import os
import sys
import time
import psutil
import logging

# Set environment variables for testing
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['PYTORCH_JIT'] = '0'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_all_imports():
    """Test importing all required modules"""
    logger.info("Testing all imports...")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Test core imports
        logger.info("Testing core imports...")
        import fastapi
        import uvicorn
        import pydantic
        logger.info("✅ Core imports successful")
        
        # Test file processing imports
        logger.info("Testing file processing imports...")
        import PyPDF2
        import fitz  # PyMuPDF
        from docx import Document
        logger.info("✅ File processing imports successful")
        
        # Test ML imports
        logger.info("Testing ML imports...")
        import torch
        import transformers
        import sklearn  # Fixed: scikit_learn -> sklearn
        import pandas
        import numpy
        logger.info("✅ ML imports successful")
        
        # Test NLP imports
        logger.info("Testing NLP imports...")
        import spacy
        logger.info("✅ NLP imports successful")
        
        # Test LLM imports
        logger.info("Testing LLM imports...")
        import openai
        logger.info("✅ LLM imports successful")
        
        # Test web client imports
        logger.info("Testing web client imports...")
        import aiohttp
        import httpx
        import redis
        import cachetools
        logger.info("✅ Web client imports successful")
        
        # Test optional imports
        logger.info("Testing optional imports...")
        try:
            import pytesseract
            logger.info("✅ pytesseract available")
        except ImportError:
            logger.warning("⚠️ pytesseract not available (optional)")
        
        try:
            import pdf2image
            logger.info("✅ pdf2image available")
        except ImportError:
            logger.warning("⚠️ pdf2image not available (optional)")
        
        try:
            from jose import jwt
            logger.info("✅ python-jose available")
        except ImportError:
            logger.warning("⚠️ python-jose not available (optional)")
        
        try:
            import passlib
            logger.info("✅ passlib available")
        except ImportError:
            logger.warning("⚠️ passlib not available (optional)")
        
        try:
            import bcrypt
            logger.info("✅ bcrypt available")
        except ImportError:
            logger.warning("⚠️ bcrypt not available (optional)")
        
        try:
            import sqlalchemy
            logger.info("✅ sqlalchemy available")
        except ImportError:
            logger.warning("⚠️ sqlalchemy not available (optional)")
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        logger.info(f"✅ All critical imports successful!")
        logger.info(f"⏱️  Import time: {end_time - start_time:.2f} seconds")
        logger.info(f"💾 Memory usage: {start_memory:.2f}MB → {end_memory:.2f}MB (+{end_memory - start_memory:.2f}MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_app_startup():
    """Test app startup"""
    logger.info("Testing app startup...")
    
    try:
        from ml_architecture.main import app
        logger.info("✅ App import successful")
        
        # Test health endpoint
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        
        if response.status_code == 200:
            logger.info("✅ Health endpoint working")
            return True
        else:
            logger.error(f"❌ Health endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ App startup failed: {e}")
        return False

def test_file_processing():
    """Test file processing capabilities"""
    logger.info("Testing file processing...")
    
    try:
        # Test PDF processing
        import PyPDF2
        logger.info("✅ PDF processing available")
        
        # Test DOCX processing
        from docx import Document
        logger.info("✅ DOCX processing available")
        
        # Test PyMuPDF
        import fitz
        logger.info("✅ PyMuPDF available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File processing test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting full deployment test...")
    
    # Test all imports
    if not test_all_imports():
        logger.error("❌ Import test failed")
        sys.exit(1)
    
    # Test app startup
    if not test_app_startup():
        logger.error("❌ App startup test failed")
        sys.exit(1)
    
    # Test file processing
    if not test_file_processing():
        logger.error("❌ File processing test failed")
        sys.exit(1)
    
    logger.info("✅ All tests passed! Ready for deployment with all dependencies.")

if __name__ == "__main__":
    main() 