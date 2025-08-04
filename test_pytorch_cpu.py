#!/usr/bin/env python3
"""
Test PyTorch CPU mode configuration
"""
import os
import torch
import logging

# Set environment variables for CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_CUDA_DEVICE'] = 'cpu'
os.environ['PYTORCH_JIT'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pytorch_cpu():
    """Test PyTorch CPU configuration"""
    logger.info("Testing PyTorch CPU configuration...")
    
    try:
        # Check PyTorch version
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        # Force CPU mode
        if cuda_available:
            logger.info("CUDA available but forcing CPU mode")
        else:
            logger.info("CUDA not available, using CPU mode")
        
        # Test tensor operations on CPU
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        
        # Create a simple tensor
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = torch.mm(x, y)
        
        logger.info(f"Tensor shape: {z.shape}")
        logger.info(f"Tensor device: {z.device}")
        
        # Test memory usage
        memory_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        logger.info(f"Memory allocated: {memory_allocated} bytes")
        
        logger.info("✅ PyTorch CPU mode working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ PyTorch CPU test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization settings"""
    logger.info("Testing memory optimization...")
    
    try:
        # Check environment variables
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        pytorch_device = os.environ.get('PYTORCH_CUDA_DEVICE', '')
        pytorch_jit = os.environ.get('PYTORCH_JIT', '')
        
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        logger.info(f"PYTORCH_CUDA_DEVICE: {pytorch_device}")
        logger.info(f"PYTORCH_JIT: {pytorch_jit}")
        
        # Test transformers
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        logger.info("✅ Memory optimization settings applied!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory optimization test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Testing PyTorch CPU configuration...")
    
    if test_pytorch_cpu() and test_memory_optimization():
        logger.info("✅ All PyTorch CPU tests passed!")
    else:
        logger.error("❌ Some PyTorch CPU tests failed!") 