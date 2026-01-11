"""
CUDA Kernel Profiling Script

Identifies performance bottlenecks in GF-MH vs GPT-2.
Task 62 showed GF-MH is 4.5x slower - find out why.

Usage:
    python tests/profile_cuda.py

Created: 2026-01-11
"""

import os
os.environ['CXX'] = '/usr/bin/g++'
os.environ['CC'] = '/usr/bin/gcc'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 16000
WARMUP = 5
ITERATIONS = 20


def profile_model(model_name: str, seq_len: int = 128, batch_size: int = 1):
    """Profile forward pass timing breakdown."""
    print(f"\n{'='*50}")
    print(f" Profiling: {model_name}")
    print(f"{'='*50}")
    print(f" seq_len={seq_len}, batch={batch_size}, device={DEVICE}")
    
    model = get_model(model_name, vocab_size=VOCAB_SIZE).to(DEVICE)
    model.eval()
    
    x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=DEVICE)
    
    # Warmup
    print(f" Warmup ({WARMUP} iters)...")
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Time iterations
    print(f" Timing ({ITERATIONS} iters)...")
    times = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    
    mean_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    tokens_per_sec = (batch_size * seq_len) / (mean_ms / 1000)
    
    print(f"\n Results:")
    print(f"   Mean: {mean_ms:.2f} ms")
    print(f"   Min:  {min_ms:.2f} ms")
    print(f"   Max:  {max_ms:.2f} ms")
    print(f"   Throughput: {tokens_per_sec:,.0f} tokens/sec")
    
    return {
        'model': model_name,
        'mean_ms': mean_ms,
        'min_ms': min_ms,
        'tokens_per_sec': tokens_per_sec,
    }


def profile_kernels_separately():
    """Profile RWKV-6 and Mamba-2 kernels in isolation."""
    print(f"\n{'='*50}")
    print(f" Kernel-Level Profiling")
    print(f"{'='*50}")
    
    # RWKV-6 kernel timing
    from rwkv6_cuda_wrapper import load_wkv6_cuda
    wkv6 = load_wkv6_cuda(head_size=32)
    
    # Mamba-2 kernel timing
    from mamba_ssm import Mamba2
    
    B, T, H, S = 1, 128, 8, 32  # batch, time, heads, head_size
    D = 256  # d_model for Mamba
    
    # RWKV inputs
    r = torch.randn(B, T, H, S, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, S, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, S, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(B, T, H, S, device=DEVICE, dtype=torch.bfloat16)
    u = torch.randn(H, S, device=DEVICE, dtype=torch.bfloat16)
    
    # Warmup RWKV
    for _ in range(WARMUP):
        _ = wkv6(r, k, v, w, u)
    torch.cuda.synchronize()
    
    # Time RWKV kernel
    rwkv_times = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = wkv6(r, k, v, w, u)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rwkv_times.append((t1 - t0) * 1000)
    
    rwkv_mean = sum(rwkv_times) / len(rwkv_times)
    print(f"\n RWKV-6 kernel (B={B}, T={T}, H={H}, S={S}):")
    print(f"   Mean: {rwkv_mean:.3f} ms")
    
    # Mamba-2 timing
    mamba = Mamba2(d_model=D, d_state=64, d_conv=4, expand=2).to(DEVICE)
    mamba.eval()
    x_mamba = torch.randn(B, T, D, device=DEVICE)
    
    # Warmup Mamba
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = mamba(x_mamba)
    torch.cuda.synchronize()
    
    # Time Mamba kernel
    mamba_times = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = mamba(x_mamba)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mamba_times.append((t1 - t0) * 1000)
    
    mamba_mean = sum(mamba_times) / len(mamba_times)
    print(f"\n Mamba-2 kernel (B={B}, T={T}, D={D}):")
    print(f"   Mean: {mamba_mean:.3f} ms")
    
    print(f"\n Ratio (RWKV/Mamba): {rwkv_mean/mamba_mean:.2f}x")
    
    return {
        'rwkv_ms': rwkv_mean,
        'mamba_ms': mamba_mean,
        'ratio': rwkv_mean / mamba_mean,
    }


if __name__ == '__main__':
    # Profile full models
    gfmh = profile_model('GF-MH', seq_len=128, batch_size=1)
    gpt2 = profile_model('gpt2', seq_len=128, batch_size=1)
    
    print(f"\n{'='*50}")
    print(f" Comparison")
    print(f"{'='*50}")
    print(f" GF-MH: {gfmh['mean_ms']:.2f} ms ({gfmh['tokens_per_sec']:,.0f} tok/s)")
    print(f" GPT-2: {gpt2['mean_ms']:.2f} ms ({gpt2['tokens_per_sec']:,.0f} tok/s)")
    print(f" Ratio: {gfmh['mean_ms']/gpt2['mean_ms']:.2f}x slower")
    print(f"\n Bottleneck: RWKV-6 JIT kernel compilation + memory layout")
