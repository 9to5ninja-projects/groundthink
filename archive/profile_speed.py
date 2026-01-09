"""
Profile where time is spent and what's actually happening.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import time

from layers import GroundThinkBlock, RMSNorm

# Tiny test
B, T, H, D = 8, 256, 8, 32  # batch, seq, heads, head_dim
dim = H * D  # 256

print(f"Config: B={B}, T={T}, H={H}, D={D}, dim={dim}")
print(f"Tokens per batch: {B * T:,}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Create one block
block = GroundThinkBlock(dim=dim, n_heads=H, head_dim=D).to(device)

# Warmup
x = torch.randn(B, T, dim, device=device)
for _ in range(3):
    out, _ = block(x)
    out.sum().backward()

torch.cuda.synchronize()

# Profile forward only
print("\n=== FORWARD ONLY ===")
x = torch.randn(B, T, dim, device=device)
torch.cuda.synchronize()
start = time.perf_counter()

for i in range(10):
    with torch.no_grad():
        out, _ = block(x)
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
tokens = B * T * 10
print(f"Time: {elapsed:.3f}s for {tokens:,} tokens")
print(f"Speed: {tokens/elapsed:,.0f} tok/s")

# Profile forward+backward
print("\n=== FORWARD + BACKWARD ===")
x = torch.randn(B, T, dim, device=device, requires_grad=True)
torch.cuda.synchronize()
start = time.perf_counter()

for i in range(10):
    out, _ = block(x)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.3f}s for {tokens:,} tokens")
print(f"Speed: {tokens/elapsed:,.0f} tok/s")

# Profile just the selective_scan
print("\n=== SELECTIVE SCAN ONLY ===")
from ops.selective_scan_triton import selective_scan_triton_forward

torch.cuda.synchronize()
start = time.perf_counter()

for i in range(10):
    k = torch.randn(B, T, H, D, device=device, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, requires_grad=True)
    w = torch.sigmoid(torch.randn(B, T, H, D, device=device))
    r = torch.randn(B, T, H, D, device=device, requires_grad=True)
    
    out = selective_scan_triton_forward(k, v, w, r)
    out.sum().backward()
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.3f}s for {tokens:,} tokens")
print(f"Speed: {tokens/elapsed:,.0f} tok/s")

# What SHOULD the speed be? Let's compare to simple matmul
print("\n=== BASELINE: SIMPLE MATMUL ===")
A = torch.randn(B * T, dim, dim, device=device)
B_mat = torch.randn(B * T, dim, 1, device=device)

torch.cuda.synchronize()
start = time.perf_counter()

for i in range(10):
    C = torch.bmm(A, B_mat)
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.3f}s for {B*T*10:,} matmuls")
print(f"If each timestep = 1 matmul, potential: {B*T*10/elapsed:,.0f} tok/s")

# The problem visualization
print("\n=== THE PROBLEM ===")
print(f"Recurrence requires T={T} SEQUENTIAL operations per layer")
print(f"With {6} layers, that's {T*6} sequential ops per batch")
print(f"Each op is a small matmul - GPU is massively underutilized")
print(f"This is why RNNs lost to Transformers for training!")
