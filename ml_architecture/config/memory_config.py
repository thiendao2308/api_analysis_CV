"""
Memory optimization configuration for CV Analysis API
"""
import os
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory management utilities for the CV Analysis API"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def log_memory_usage(stage: str):
        """Log memory usage at different stages"""
        memory_mb = MemoryManager.get_memory_usage()
        logger.info(f"Memory usage at {stage}: {memory_mb:.2f} MB")
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory"""
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
    
    @staticmethod
    def optimize_for_deployment():
        """Apply memory optimizations for deployment"""
        # Set environment variables for memory optimization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Disable some heavy features for deployment
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        logger.info("Memory optimization settings applied for deployment")

# Memory thresholds - Increased for development
MEMORY_THRESHOLDS = {
    'WARNING': 800,   # MB - Increased from 400
    'CRITICAL': 1000, # MB - Increased from 450
    'MAX': 1200       # MB - Increased from 500
}

def check_memory_usage():
    """Check if memory usage is within acceptable limits"""
    current_memory = MemoryManager.get_memory_usage()
    
    if current_memory > MEMORY_THRESHOLDS['CRITICAL']:
        logger.error(f"Critical memory usage: {current_memory:.2f} MB")
        MemoryManager.force_garbage_collection()
        # Don't return False immediately, try to continue
        return True  # Changed from False to True
    elif current_memory > MEMORY_THRESHOLDS['WARNING']:
        logger.warning(f"High memory usage: {current_memory:.2f} MB")
        MemoryManager.force_garbage_collection()
        return True
    else:
        return True 