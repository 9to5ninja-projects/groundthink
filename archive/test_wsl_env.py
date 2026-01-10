#!/usr/bin/env python3
"""Test that WSL2 environment works for training."""

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Test FLA imports
try:
    from fla.layers.rwkv6 import RWKV6Attention
    print("RWKV6Attention: OK")
except Exception as e:
    print(f"RWKV6Attention: FAILED - {e}")

try:
    from fla.layers.mamba2 import Mamba2
    print("Mamba2: OK")
except Exception as e:
    print(f"Mamba2: FAILED - {e}")

# Test hybrid model
try:
    from hybrid_v4 import create_hybrid_5m
    model = create_hybrid_5m(vocab_size=100)
    model = model.cuda()
    x = torch.randint(0, 100, (2, 64)).cuda()
    with torch.no_grad():
        out = model(x)
    print(f"Hybrid model forward: OK - output shape {out.shape}")
except Exception as e:
    import traceback
    print(f"Hybrid model: FAILED - {e}")
    traceback.print_exc()

print("\n=== Environment Ready ===")
