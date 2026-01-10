"""
FLA Replacements - Bridge to CUDA Implementations

Provides RWKV6Attention and Mamba2 classes for hybrid_v4.py
Uses CUDA kernels when available, falls back to PyTorch.
"""

import os

# Set compiler environment variables before importing torch
# This fixes "g++-12 not found" issue on systems with older g++
if 'CXX' not in os.environ:
    os.environ['CXX'] = '/usr/bin/g++'
if 'CC' not in os.environ:
    os.environ['CC'] = '/usr/bin/gcc'

import torch
import torch.nn as nn

# Try CUDA wrapper first, fall back to prototype
try:
    from rwkv6_cuda_wrapper import RWKV6Attention_CUDA
    RWKV6_IMPL = RWKV6Attention_CUDA
    RWKV6_CUDA_AVAILABLE = True
except ImportError:
    from rwkv6_prototype import RWKV6Attention_Prototype
    RWKV6_IMPL = RWKV6Attention_Prototype
    RWKV6_CUDA_AVAILABLE = False

# Import Mamba-2 from mamba-ssm (already has CUDA)
from mamba_ssm import Mamba2 as Mamba2_SSM


class RWKV6Attention(nn.Module):
    """RWKV-6 Attention wrapper for hybrid_v4.py compatibility"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, layer_idx: int = 0):
        super().__init__()
        head_size = hidden_size // num_heads
        
        if RWKV6_CUDA_AVAILABLE:
            # Use CUDA wrapper
            self.rwkv = RWKV6_IMPL(
                hidden_size=hidden_size,
                n_head=num_heads,
                head_size=head_size,
            )
            self._using_cuda = True
        else:
            # Use prototype
            self.rwkv = RWKV6_IMPL(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_size=head_size,
            )
            self._using_cuda = False
    
    def forward(self, x, attention_mask=None, past_key_values=None):
        """Returns (output, None, None) for compatibility"""
        result = self.rwkv(x)
        # Handle tuple output from prototype
        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result
        return out, None, None


class Mamba2(nn.Module):
    """Mamba-2 wrapper for hybrid_v4.py compatibility"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, head_dim: int = 64,
                 expand: int = 2, layer_idx: int = 0):
        super().__init__()
        self.mamba = Mamba2_SSM(
            d_model=hidden_size,
            d_state=64,
            d_conv=4,
            expand=expand,
            headdim=head_dim,
        )
    
    def forward(self, x):
        return self.mamba(x)


if __name__ == "__main__":
    print("Testing fla_replacements...")
    print(f"RWKV-6 using CUDA: {RWKV6_CUDA_AVAILABLE}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test RWKV6Attention
    rwkv = RWKV6Attention(hidden_size=128, num_heads=4).to(device)
    x = torch.randn(2, 32, 128, device=device)
    out_rwkv, _, _ = rwkv(x)
    print(f"RWKV6: {x.shape} -> {out_rwkv.shape}")
    print(f"  Using CUDA wrapper: {getattr(rwkv, '_using_cuda', False)}")
    
    # Test Mamba2
    mamba = Mamba2(hidden_size=128, num_heads=4).to(device)
    out = mamba(x)
    print(f"Mamba2: {x.shape} -> {out.shape}")
    
    print("✓ Both components working!")
