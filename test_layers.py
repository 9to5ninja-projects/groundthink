"""Quick test for updated layers with selective scan kernel."""
import sys
sys.path.insert(0, '/workspace/groundthink')

import torch
from layers import TimeMixing
import torch.nn as nn

# Test TimeMixing directly
print("Testing TimeMixing layer...")
tm = TimeMixing(dim=128, n_heads=2, head_dim=64).cuda()

# Check out_proj initialization
print(f"out_proj.weight stats: min={tm.out_proj.weight.min():.6f}, max={tm.out_proj.weight.max():.6f}")
print(f"out_proj is all zeros: {(tm.out_proj.weight == 0).all()}")

# Initialize out_proj with small random values instead of zeros
nn.init.normal_(tm.out_proj.weight, std=0.02)
print(f"After re-init: out_proj.weight stats: min={tm.out_proj.weight.min():.6f}, max={tm.out_proj.weight.max():.6f}")

# Use float32 for stability
x = torch.randn(2, 32, 128, dtype=torch.float32).cuda()
x.requires_grad_(True)
x.retain_grad()

out, state = tm(x)
print(f"Forward OK: output shape = {out.shape}")
print(f"Output stats: min={out.min().item():.4f}, max={out.max().item():.4f}")

loss = out.mean()
loss.backward()

print(f"Backward OK: x.grad exists = {x.grad is not None}")

# Check gradients on layer parameters
for name, p in tm.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        print(f"  {name}: norm={gn:.6f}")
