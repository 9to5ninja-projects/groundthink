"""
Environment initialization for GroundThink

IMPORT THIS FIRST in any script that uses PyTorch + NumPy on Windows.

CRITICAL REQUIREMENTS FOR FLA/TRITON:
1. KMP_DUPLICATE_LIB_OK=TRUE - Fixes OpenMP conflict between conda NumPy (Intel MKL) and PyTorch
2. ALL TENSORS MUST BE ON CUDA - FLA uses Triton kernels which require GPU tensors
   - Model must be .to(device) where device = "cuda"
   - Input tensors must be created with device=device
   - CPU tensors will crash with "Pointer argument cannot be accessed from Triton (cpu tensor?)"

Usage:
    import env_init  # MUST be first import
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MyModel().to(device)
    x = torch.randint(0, vocab_size, (batch, seq), device=device)
    out = model(x)
"""
import os

# Fix OpenMP conflict (Intel MKL vs PyTorch)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Optional: Set CUDA device visibility if needed
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
