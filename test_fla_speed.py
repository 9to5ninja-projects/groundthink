"""
Test FLA simple_gla speed vs our implementation.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.insert(0, 'e:/RWKV/fla')

import torch
import time

device = 'cuda'
B, T, H, D = 8, 256, 8, 32

print(f"Config: B={B}, T={T}, H={H}, D={D}")
print(f"Tokens per batch: {B * T:,}")

# Test FLA simple_gla
print("\n=== FLA chunk_simple_gla ===")
try:
    print("Importing FLA...")
    from fla.ops.simple_gla import chunk_simple_gla
    print("Import OK. Creating tensors...")
    
    # FLA expects: q, k, v: [B, T, H, D], g: [B, T, H] (scalar per head per timestep)
    # Using head_first=False (default)
    q = torch.randn(B, T, H, D, device=device, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, requires_grad=True)
    g = torch.randn(B, T, H, device=device, requires_grad=True)  # Log-space gate
    print("Tensors OK. Running warmup (compiling Triton kernels - may take 1-2 min first time)...")
    
    # Warmup - just 1 iteration since compilation is slow
    o, _ = chunk_simple_gla(q, k, v, g)
    print("Forward done. Running backward...")
    o.sum().backward()
    q.grad = k.grad = v.grad = g.grad = None
    print("Warmup done. Benchmarking...")
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(10):
        q.grad = k.grad = v.grad = g.grad = None
        o, _ = chunk_simple_gla(q, k, v, g)
        o.sum().backward()
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    tokens = B * T * 10
    print(f"Time: {elapsed:.3f}s for {tokens:,} tokens")
    print(f"Speed: {tokens/elapsed:,.0f} tok/s")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Compare to our implementation
print("\n=== Our selective_scan ===")
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
tokens = B * T * 10
print(f"Time: {elapsed:.3f}s for {tokens:,} tokens")
print(f"Speed: {tokens/elapsed:,.0f} tok/s")

print("\n=== Summary ===")
print("FLA uses chunked parallel computation - should be 10-50x faster")
