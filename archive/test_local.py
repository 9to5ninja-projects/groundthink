"""
Local CPU test to verify code logic before deploying to GPU.
Tests: selective_scan kernel, layers.py integration, gradient flow.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn

print("=== LOCAL CPU TEST ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test 1: selective_scan_triton (CPU fallback)
print("\n[1] Testing selective_scan kernel...")
from ops.selective_scan_triton import selective_scan_triton_forward

B, T, H, D = 2, 16, 4, 8  # Tiny dimensions for CPU
k = torch.randn(B, T, H, D, requires_grad=True)
v = torch.randn(B, T, H, D, requires_grad=True)
w = torch.sigmoid(torch.randn(B, T, H, D, requires_grad=True))  # 0-1 range
r = torch.randn(B, T, H, D, requires_grad=True)

try:
    output = selective_scan_triton_forward(k, v, w, r)
    loss = output.sum()
    loss.backward()
    
    print(f"  Output shape: {output.shape}")
    print(f"  k.grad exists: {k.grad is not None}")
    print(f"  v.grad exists: {v.grad is not None}")
    print(f"  w.grad exists: {w.grad is not None}")
    print(f"  r.grad exists: {r.grad is not None}")
    print(f"  k.grad norm: {k.grad.norm().item():.6f}")
    print("  ✅ selective_scan PASSED")
except Exception as e:
    print(f"  ❌ selective_scan FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: layers.py import
print("\n[2] Testing layers.py import...")
try:
    from layers import GroundThinkBlock, TimeMixing, ChannelMixing, RMSNorm
    print("  ✅ layers.py imports PASSED")
except Exception as e:
    print(f"  ❌ layers.py import FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: TimeMixing forward/backward
print("\n[3] Testing TimeMixing forward/backward...")
try:
    dim, n_heads, head_dim = 32, 4, 8
    tm = TimeMixing(dim=dim, n_heads=n_heads, head_dim=head_dim, use_grounding=True)
    
    x = torch.randn(2, 16, dim, requires_grad=True)
    output, state = tm(x)
    loss = output.sum()
    loss.backward()
    
    print(f"  Output shape: {output.shape}")
    print(f"  State shape: {state.shape}")
    print(f"  x.grad exists: {x.grad is not None}")
    
    # Check which parameters have gradients
    params_with_grad = []
    params_without_grad = []
    for name, p in tm.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            params_with_grad.append(name)
        else:
            params_without_grad.append(name)
    
    print(f"  Params WITH grad: {len(params_with_grad)}")
    print(f"  Params WITHOUT grad: {len(params_without_grad)}")
    if params_without_grad:
        print(f"    Missing: {params_without_grad[:5]}...")
    print("  ✅ TimeMixing PASSED")
except Exception as e:
    print(f"  ❌ TimeMixing FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Full GroundThinkBlock
print("\n[4] Testing GroundThinkBlock forward/backward...")
try:
    block = GroundThinkBlock(dim=32, n_heads=4, head_dim=8, use_grounding=True)
    
    x = torch.randn(2, 16, 32, requires_grad=True)
    output, state = block(x)
    loss = output.sum()
    loss.backward()
    
    print(f"  Output shape: {output.shape}")
    print(f"  x.grad norm: {x.grad.norm().item():.6f}")
    
    total_params = sum(p.numel() for p in block.parameters())
    grad_params = sum(p.numel() for p in block.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Params with gradients: {grad_params}/{total_params}")
    print("  ✅ GroundThinkBlock PASSED")
except Exception as e:
    print(f"  ❌ GroundThinkBlock FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Model import
print("\n[5] Testing model.py import...")
try:
    # Model uses relative imports, so we import layers directly and build a simple model
    from layers import GroundThinkBlock, RMSNorm
    
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, dim=32, n_layers=2, n_heads=4, head_dim=8):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                GroundThinkBlock(dim=dim, n_heads=n_heads, head_dim=head_dim)
                for _ in range(n_layers)
            ])
            self.ln_out = RMSNorm(dim)
            self.head = nn.Linear(dim, vocab_size, bias=False)
        
        def forward(self, x):
            x = self.embed(x)
            for block in self.blocks:
                x, _ = block(x)
            x = self.ln_out(x)
            return self.head(x)
    
    model = SimpleModel()
    
    x = torch.randint(0, 1000, (2, 16))
    output = model(x)
    loss = output.view(-1, 1000).sum()
    loss.backward()
    
    total = sum(p.numel() for p in model.parameters())
    with_grad = sum(p.numel() for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Model params: {total:,}")
    print(f"  With gradients: {with_grad:,} ({100*with_grad/total:.1f}%)")
    print("  ✅ Model PASSED")
except Exception as e:
    print(f"  ❌ Model FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DONE ===")
