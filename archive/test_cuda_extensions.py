# test_cuda_extensions.py
import torch
import sys
import os

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

# Test causal-conv1d
try:
    import causal_conv1d
    print("✓ causal_conv1d imported")
    
    # Try to access CUDA function
    from causal_conv1d import causal_conv1d_fn
    if causal_conv1d_fn is not None:
        print("✓ causal_conv1d CUDA kernel available")
    else:
        print("✗ causal_conv1d CUDA kernel is None")
except Exception as e:
    print(f"✗ causal_conv1d import failed: {e}")

# Test mamba-ssm
try:
    import mamba_ssm
    print("✓ mamba_ssm imported")
    
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    if selective_scan_fn is not None:
        print("✓ mamba_ssm selective_scan_fn available")
    else:
        print("✗ selective_scan_fn is None")
except Exception as e:
    print(f"✗ mamba_ssm import failed: {e}")

# Test FLA
try:
    import fla
    print(f"✓ FLA imported (version: {fla.__version__})")
except Exception as e:
    print(f"✗ FLA import failed: {e}")

# Test actual CUDA compilation
if torch.cuda.is_available():
    x = torch.randn(2, 32, 256, device='cuda')
    print(f"✓ Tensor created on CUDA: {x.device}")
    
    # Try a simple CUDA operation
    y = x @ x.transpose(-1, -2)
    print(f"✓ CUDA matrix multiplication works")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
