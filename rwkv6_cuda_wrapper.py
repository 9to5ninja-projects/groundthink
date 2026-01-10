"""
RWKV-6 CUDA Kernel Wrapper

Purpose: Integrate official RWKV-CUDA kernels with PyTorch for production speed.
Falls back to PyTorch prototype when CUDA kernel unavailable or incompatible.

Key Features:
- JIT compilation of wkv6 CUDA kernel
- Automatic dtype conversion (model fp32/fp16 → kernel bf16)
- Fallback to RWKV6Attention_Prototype when needed
- Compatible with hybrid_v4.py interface

Requirements:
- CUDA toolkit 11.8+
- ninja (for JIT compilation)
- RWKV-CUDA/wkv6/ kernel sources

Reference: V4.5_CUDA_KERNELS.md, RWKV-CUDA/wkv6/run.py
Created: 2026-01-09
"""

import os

# Set compiler environment variables before importing torch
# This fixes "g++-12 not found" issue on systems with older g++
# MUST use absolute paths and set unconditionally
os.environ['CXX'] = '/usr/bin/g++'
os.environ['CC'] = '/usr/bin/gcc'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Import prototype for fallback
from rwkv6_prototype import RWKV6Attention_Prototype


class WKV6_CUDA(torch.autograd.Function):
    """
    Autograd function wrapping the WKV6 CUDA kernel.
    
    Kernel interface: forward(B, T, C, H, r, k, v, w, u, y)
    - B: batch size
    - T: sequence length
    - C: channels (hidden_size)
    - H: number of heads
    - r, k, v: receptance, key, value tensors [B, T, C] in bf16
    - w: time decay (already exp-transformed) [B, T, C] in float32
    - u: time_first parameter [H, C//H] in bf16
    - y: output tensor [B, T, C] in bf16
    """
    
    CUDA_KERNEL = None  # Will be set by RWKV6Attention_CUDA
    
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            # Ensure correct dtypes
            assert r.dtype == torch.bfloat16, f"r must be bf16, got {r.dtype}"
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            
            # Ensure contiguous
            r = r.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            w = w.contiguous()
            u = u.contiguous()
            
            # Transform w for kernel: exp(-exp(w))
            ew = (-torch.exp(w.float())).contiguous()
            
            # Save for backward
            ctx.B, ctx.T, ctx.C, ctx.H = B, T, C, H
            ctx.save_for_backward(r, k, v, ew, u)
            
            # Allocate output
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16,
                          memory_format=torch.contiguous_format)
            
            # Call CUDA kernel
            WKV6_CUDA.CUDA_KERNEL.forward(B, T, C, H, r, k, v, ew, u, y)
            
            return y
    
    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            gy = gy.contiguous()
            B, T, C, H = ctx.B, ctx.T, ctx.C, ctx.H
            r, k, v, ew, u = ctx.saved_tensors
            
            # Allocate gradient tensors
            gr = torch.empty_like(r)
            gk = torch.empty_like(k)
            gv = torch.empty_like(v)
            gw = torch.zeros((B, T, C), device=gy.device, dtype=torch.bfloat16).contiguous()
            gu = torch.empty((B, C), device=gy.device, dtype=torch.bfloat16,
                           memory_format=torch.contiguous_format)
            
            # Call CUDA backward kernel
            WKV6_CUDA.CUDA_KERNEL.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            
            # Reduce gu across batch
            gu = torch.sum(gu, 0).view(H, C // H)
            
            return (None, None, None, None, gr, gk, gv, gw, gu)


class RWKV6Attention_CUDA(nn.Module):
    """
    Production RWKV-6 with CUDA kernels.
    
    Falls back to PyTorch prototype when:
    - CUDA kernel compilation fails
    - Input not on GPU
    - Dtype/shape incompatibility
    
    Args:
        hidden_size: Model dimension
        n_head: Number of attention heads (default: computed from hidden_size/64)
        head_size: Dimension per head (default: 64)
    """
    
    # Class-level kernel cache (compiled once per head_size/seq_len combo)
    _kernel_cache = {}
    
    def __init__(self, hidden_size, n_head=None, head_size=64, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        
        if n_head is None:
            n_head = hidden_size // head_size
        self.n_head = n_head
        
        # Validate dimensions
        assert hidden_size == n_head * head_size, \
            f"hidden_size ({hidden_size}) must equal n_head ({n_head}) * head_size ({head_size})"
        
        # Time-mixing parameters (matches prototype)
        self.time_decay = nn.Parameter(torch.ones(n_head, head_size) * -5.0)
        self.time_first = nn.Parameter(torch.zeros(n_head, head_size))
        
        # Linear projections (no bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Channel-mixing (FFN)
        self.ffn_key = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.ffn_value = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Fallback prototype
        self._prototype = None
        
        # Try to compile CUDA kernel
        self._cuda_available = self._try_compile_kernel()
        
    def _try_compile_kernel(self):
        """Attempt to JIT compile the CUDA kernel."""
        cache_key = self.head_size
        
        if cache_key in RWKV6Attention_CUDA._kernel_cache:
            WKV6_CUDA.CUDA_KERNEL = RWKV6Attention_CUDA._kernel_cache[cache_key]
            return True
        
        # Find kernel source directory
        kernel_dir = os.path.join(os.path.dirname(__file__), 'RWKV-CUDA', 'wkv6', 'cuda')
        
        if not os.path.exists(kernel_dir):
            print(f"RWKV-CUDA kernel directory not found: {kernel_dir}")
            print("Falling back to PyTorch prototype")
            return False
        
        try:
            # Note: T (seq_len) is set at compile time. We use a reasonable default.
            # For production, you may need to recompile for different seq lengths.
            default_seq_len = 256
            
            cuda_kernel = load(
                name=f"wkv6_h{self.head_size}",
                sources=[
                    os.path.join(kernel_dir, "wkv6_op.cpp"),
                    os.path.join(kernel_dir, "wkv6_cuda_v1.cu"),
                ],
                verbose=True,
                extra_cuda_cflags=[
                    "-res-usage",
                    "--use_fast_math",
                    "-O3",
                    "-Xptxas", "-O3",
                    "--extra-device-vectorization",
                    f"-D_N_={self.head_size}",
                    f"-D_T_={default_seq_len}",
                ]
            )
            
            RWKV6Attention_CUDA._kernel_cache[cache_key] = cuda_kernel
            WKV6_CUDA.CUDA_KERNEL = cuda_kernel
            print(f"✓ RWKV-6 CUDA kernel compiled (head_size={self.head_size})")
            return True
            
        except Exception as e:
            print(f"CUDA kernel compilation failed: {e}")
            print("Falling back to PyTorch prototype")
            return False
    
    def _get_prototype(self):
        """Lazy initialization of fallback prototype."""
        if self._prototype is None:
            self._prototype = RWKV6Attention_Prototype(
                self.hidden_size, 
                n_head=self.n_head,
                head_size=self.head_size
            )
            # Copy current parameters to prototype
            self._prototype.time_decay.data = self.time_decay.data.clone()
            self._prototype.time_first.data = self.time_first.data.clone()
            self._prototype.key.weight.data = self.key.weight.data.clone()
            self._prototype.value.weight.data = self.value.weight.data.clone()
            self._prototype.receptance.weight.data = self.receptance.weight.data.clone()
            self._prototype.output.weight.data = self.output.weight.data.clone()
            self._prototype.ffn_key.weight.data = self.ffn_key.weight.data.clone()
            self._prototype.ffn_value.weight.data = self.ffn_value.weight.data.clone()
            self._prototype.ln1.weight.data = self.ln1.weight.data.clone()
            self._prototype.ln1.bias.data = self.ln1.bias.data.clone()
            self._prototype.ln2.weight.data = self.ln2.weight.data.clone()
            self._prototype.ln2.bias.data = self.ln2.bias.data.clone()
            self._prototype = self._prototype.to(self.time_decay.device)
        return self._prototype
    
    def forward(self, x, state=None):
        """
        Forward pass with automatic CUDA/PyTorch selection.
        
        Args:
            x: Input tensor [B, T, C]
            state: Optional recurrent state (unused)
            
        Returns:
            Tuple of (output, None, None)
        """
        B, T, C = x.shape
        H, S = self.n_head, self.head_size
        
        # Use prototype fallback if CUDA not available or CPU input
        if not self._cuda_available or not x.is_cuda:
            return self._get_prototype()(x, state)
        
        try:
            # ========== Time-Mixing Block ==========
            x_ln = self.ln1(x)
            
            # Project to r, k, v and convert to bf16
            r = self.receptance(x_ln).bfloat16().contiguous()  # [B, T, C]
            k = self.key(x_ln).bfloat16().contiguous()
            v = self.value(x_ln).bfloat16().contiguous()
            
            # Time parameters - expand to [B, T, C] for kernel
            # w: [H, S] → [1, 1, C] → [B, T, C]
            w = self.time_decay.view(1, 1, C).expand(B, T, C).bfloat16().contiguous()
            u = self.time_first.bfloat16().contiguous()  # [H, S]
            
            # Call CUDA kernel via autograd function
            wkv = WKV6_CUDA.apply(B, T, C, H, r, k, v, w, u)
            
            # Apply receptance gating (sigmoid already applied in kernel output usage)
            rwkv = torch.sigmoid(r.float()) * wkv.float()
            output1 = self.output(rwkv.view(B, T, C).to(x.dtype))
            
            # Residual
            x = x + output1
            
            # ========== Channel-Mixing Block ==========
            x_ln = self.ln2(x)
            ffn_out = self.ffn_value(F.relu(self.ffn_key(x_ln)) ** 2)
            x = x + ffn_out
            
            return x, None, None
            
        except Exception as e:
            print(f"CUDA kernel failed: {e}, falling back to prototype")
            return self._get_prototype()(x, state)


# Alias for compatibility
RWKV6Attention = RWKV6Attention_CUDA


def test_rwkv6_cuda():
    """Test CUDA kernel wrapper."""
    print("Testing RWKV6Attention_CUDA...")
    
    hidden_size = 256
    model = RWKV6Attention_CUDA(hidden_size, n_head=4, head_size=64)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(2, 32, hidden_size).cuda()
        
        output, _, _ = model(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        assert not torch.isnan(output).any(), "NaN detected in output"
        
        print(f"  ✓ Forward pass: {x.shape} -> {output.shape}")
        print(f"  ✓ No NaN: True")
        print(f"  ✓ CUDA kernel used: {model._cuda_available}")
    else:
        print("  CUDA not available, skipping test")
    
    print("RWKV6Attention_CUDA: Test complete!")
    return True


if __name__ == "__main__":
    test_rwkv6_cuda()
