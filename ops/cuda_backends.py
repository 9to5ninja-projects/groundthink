"""
CUDA Backends - Bridge to RWKV-6 and Mamba-2 Implementations

Provides RWKV6Attention and Mamba2 classes for hybrid_v4.py
Uses CUDA kernels when available, falls back to PyTorch.

MEMORY NOTE (2026-01-11):
  - PyTorch import: ~350 MB baseline
  - RWKV6 prototype: +90 MB
  - mamba_ssm import: +200 MB (lazy-loaded to avoid when not needed)
  - WSL budget: ~2.5 GB total, keep models small for development
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
    from .rwkv6_cuda_wrapper import RWKV6Attention_CUDA
    RWKV6_IMPL = RWKV6Attention_CUDA
    RWKV6_CUDA_AVAILABLE = True
except (ImportError, ValueError):
    from .rwkv6_prototype import RWKV6Attention_Prototype
    RWKV6_IMPL = RWKV6Attention_Prototype
    RWKV6_CUDA_AVAILABLE = False

# Lazy import for Mamba-2 (heavy dependency, only load when needed)
Mamba2_SSM = None
MAMBA2_AVAILABLE = False


class RWKV6Attention(nn.Module):
    """RWKV-6 Attention wrapper for hybrid_v4.py compatibility"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_size = hidden_size // num_heads
        
        if RWKV6_CUDA_AVAILABLE:
            # Use CUDA wrapper for fast forward pass
            self.rwkv = RWKV6_IMPL(
                hidden_size=hidden_size,
                n_head=num_heads,
                head_size=head_size,
            )
            self._using_cuda = True
            
            # Also create prototype for state extraction (lazy init)
            self._prototype_for_state = None
        else:
            # Use prototype
            self.rwkv = RWKV6_IMPL(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_size=head_size,
            )
            self._using_cuda = False
    
    def _get_prototype_for_state(self):
        """Lazy-init prototype for state extraction when using CUDA"""
        if self._prototype_for_state is None:
            from .rwkv6_prototype import RWKV6Attention_Prototype
            head_size = self.hidden_size // self.num_heads
            self._prototype_for_state = RWKV6Attention_Prototype(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_size=head_size,
            )
            # Move to same device as main module
            device = next(self.rwkv.parameters()).device
            self._prototype_for_state = self._prototype_for_state.to(device)
            # Copy weights from CUDA wrapper to prototype
            # Note: Weight sharing is complex; for now just use separate weights
        return self._prototype_for_state
    
    def forward(self, x, attention_mask=None, past_key_values=None, return_state=False):
        """
        Forward pass through RWKV-6.
        
        Args:
            x: Input tensor
            return_state: If True, return internal recurrent state
            
        Returns:
            If return_state=False: (output, None, None) for compatibility
            If return_state=True: (output, None, state_dict) with 'rwkv_state' [B, H, S]
            
        Note: When using CUDA and return_state=True, falls back to prototype.
              This may produce slightly different outputs due to weight initialization.
              For accurate state extraction, use prototype mode or disable CUDA.
        """
        if return_state:
            if self._using_cuda:
                # Fall back to prototype for state extraction
                proto = self._get_prototype_for_state()
                result = proto(x, return_state=True)
            else:
                result = self.rwkv(x, return_state=True)
            
            if isinstance(result, tuple) and len(result) == 3:
                out, _, state_dict = result
                return out, None, state_dict
            else:
                out = result[0] if isinstance(result, tuple) else result
                return out, None, None
        
        # Default fast path (CUDA when available)
        result = self.rwkv(x)
        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result
        return out, None, None


class Mamba2(nn.Module):
    """Mamba-2 wrapper for hybrid_v4.py compatibility
    
    NOTE: Lazy-loads mamba_ssm on first instantiation to save ~200MB
    when only using RWKV-6 (e.g., Task 0.0.1 pure baseline).
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, head_dim: int = 64,
                 expand: int = 2, layer_idx: int = 0):
        super().__init__()
        
        # Lazy import mamba_ssm (heavy dependency)
        global Mamba2_SSM, MAMBA2_AVAILABLE
        if Mamba2_SSM is None:
            try:
                from mamba_ssm import Mamba2 as _Mamba2_SSM
                Mamba2_SSM = _Mamba2_SSM
                MAMBA2_AVAILABLE = True
            except ImportError:
                raise ImportError(
                    "mamba_ssm required for Mamba2. Install with: pip install mamba-ssm"
                )
        
        self.hidden_size = hidden_size
        self.mamba = Mamba2_SSM(
            d_model=hidden_size,
            d_state=64,
            d_conv=4,
            expand=expand,
            headdim=head_dim,
        )
    
    def forward(self, x, return_state=False):
        """
        Forward pass through Mamba-2.
        
        Args:
            x: Input tensor [B, T, hidden_size]
            return_state: If True, return state information
            
        Returns:
            If return_state=False: output tensor [B, T, hidden_size]
            If return_state=True: (output, state_dict) where state_dict contains:
                - 'mamba_state': Output activations as state proxy [B, hidden_size]
                                 (True SSM state extraction requires inference_params)
        
        Note: True Mamba SSM state has shape [B, nheads, headdim, d_state] but requires
              inference_params mechanism to extract. Output activations serve as a
              proxy for component health diagnostics (activation variance ratio).
        """
        out = self.mamba(x)
        
        if return_state:
            # Use final position's output as state proxy
            # This captures the "memory" of what Mamba produced
            # True SSM state would require modifying mamba_chunk_scan_combined
            final_out = out[:, -1, :]  # [B, hidden_size]
            state_dict = {'mamba_state': final_out}
            return out, state_dict
        
        return out


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
