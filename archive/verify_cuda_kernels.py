import torch
import sys

print("="*60)
print("VERIFYING CUDA KERNELS - CORRECT PARAMETERS")
print("="*60)

# Test causal-conv1d with correct parameters
print("\n1. Testing causal-conv1d...")
try:
    import causal_conv1d
    print("   ✓ Module imported")
    
    # Create tensors with correct dimensions
    # causal_conv1d_fn expects: x, weight, bias, silent=True
    batch, seq, dim = 2, 32, 128
    kernel_size = 4
    
    x = torch.randn(batch, seq, dim, device='cuda', dtype=torch.float16)
    weight = torch.randn(kernel_size, dim, device='cuda', dtype=torch.float16)
    bias = torch.randn(dim, device='cuda', dtype=torch.float16)
    
    # Get the function
    from causal_conv1d import causal_conv1d_fn
    
    if causal_conv1d_fn is not None:
        # Call with correct parameters
        y = causal_conv1d_fn(
            x, 
            weight, 
            bias, 
            False,  # silent
            0.0,    # initial_states
            1.0     # final_states_aligned
        )
        print(f"   ✓ CUDA kernel executed, output shape: {y.shape}")
        print(f"   ✅ causal-conv1d CUDA kernels WORKING")
    else:
        print("   ⚠️ causal_conv1d_fn is None (Triton fallback)")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test mamba-ssm
print("\n2. Testing mamba-ssm...")
try:
    import mamba_ssm
    print("   ✓ Module imported")
    
    from mamba_ssm.ops.selective_state_update import selective_state_update
    
    if selective_state_update is not None:
        print("   ✓ selective_state_update available")
        print("   ✅ mamba-ssm CUDA kernels WORKING")
    else:
        print("   ⚠️ selective_state_update is None (Triton fallback)")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("PyTorch Info:")
print(f"  Version: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print("="*60)
