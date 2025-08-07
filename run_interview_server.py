#!/usr/bin/env python3
"""
Run Interview System Server
Tương tự như ml_architecture, chạy server với uvicorn
"""

import os
import sys
import subprocess
import uvicorn

def run_interview_server():
    """Chạy Interview System server"""
    print("🚀 Starting Interview System Server...")
    print("=" * 50)
    
    # Set environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    
    print(f"🌐 Server will run on: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/api/v1/health")
    print("=" * 50)
    
    try:
        # Run uvicorn server
        uvicorn.run(
            "interview_system.main:app",
            host=host,
            port=port,
            reload=True,
            workers=1,
            loop="asyncio",
            log_level="info",
            timeout_keep_alive=15,
            timeout_graceful_shutdown=5,
            access_log=True,
            limit_concurrency=5,
            limit_max_requests=500,
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_interview_server() 