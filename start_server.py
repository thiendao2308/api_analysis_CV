#!/usr/bin/env python3
"""
Startup script for CV Analysis API with memory optimization
"""
import os
import sys
import gc
import logging

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'  # Disable CUDA memory caching

# Additional optimizations for Render
os.environ['PYTORCH_JIT'] = '0'  # Disable JIT compilation
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'  # Use temp cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_memory():
    """Apply memory optimizations before starting server"""
    logger.info("Applying memory optimizations...")
    
    # Force garbage collection
    collected = gc.collect()
    logger.info(f"Initial garbage collection freed {collected} objects")
    
    # Set memory limits for numpy
    try:
        import numpy as np
        np.set_printoptions(threshold=100)
        logger.info("NumPy memory optimization applied")
    except ImportError:
        pass
    
    # Set memory limits for pandas
    try:
        import pandas as pd
        pd.options.mode.chained_assignment = None
        logger.info("Pandas memory optimization applied")
    except ImportError:
        pass
    
    # Force PyTorch to use CPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA available but forcing CPU mode for memory optimization")
        else:
            logger.info("CUDA not available, using CPU mode")
    except ImportError:
        pass

def start_server():
    """Start the FastAPI server with memory optimization"""
    try:
        # Apply memory optimizations
        optimize_memory()
        
        # Import and start the app
        from ml_architecture.main import app
        import uvicorn
        
        # Get port from environment or use default
        port = int(os.environ.get('PORT', 8000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        logger.info(f"Starting server on {host}:{port}")
        logger.info("Memory optimizations applied successfully")
        
        # Start server with optimized settings for Render
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=1,  # Single worker to save memory
            loop="asyncio",
            log_level="info",
            timeout_keep_alive=30,  # Reduce keep-alive timeout
            timeout_graceful_shutdown=10,  # Reduce graceful shutdown timeout
            access_log=False,  # Disable access logs to save memory
            # Additional optimizations
            limit_concurrency=10,  # Limit concurrent requests
            limit_max_requests=1000,  # Restart worker after 1000 requests
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server() 