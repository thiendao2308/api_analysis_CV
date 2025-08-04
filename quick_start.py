#!/usr/bin/env python3
"""
Quick startup script for Render deployment
"""
import os
import sys
import logging

# Set environment variables for fast startup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['PYTORCH_JIT'] = '0'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_start():
    """Quick startup with minimal initialization"""
    try:
        logger.info("🚀 Quick startup for Render...")
        
        # Import app with minimal overhead
        from ml_architecture.main import app
        import uvicorn
        
        # Get port from environment
        port = int(os.environ.get('PORT', 8000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        logger.info(f"Starting server on {host}:{port}")
        
        # Start with minimal settings
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=1,
            loop="asyncio",
            log_level="warning",  # Reduce logging
            timeout_keep_alive=10,  # Very short timeout
            timeout_graceful_shutdown=3,  # Very short shutdown
            access_log=False,
            limit_concurrency=3,  # Very low concurrency
            limit_max_requests=100,  # Restart frequently
        )
        
    except Exception as e:
        logger.error(f"Quick startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    quick_start() 