"""
Minimal Smoke Test for WSL Memory Constraints

Purpose: Verify we can run RWKV-6 training at all before larger experiments.
Strategy: 
  - Skip mamba_ssm import (not needed for pure RWKV-6)
  - Use prototype only (no CUDA compilation)
  - Tiny batch/seq: batch=1, seq=32
  - Only 5 training steps
  - Report memory at each stage
"""

import os
import sys
import gc

def get_memory_mb():
    """Get current process memory in MB"""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"[Stage 0] Before imports: {get_memory_mb():.1f} MB")

# Minimal imports - avoid mamba_ssm
import torch
import torch.nn as nn
print(f"[Stage 1] After torch import: {get_memory_mb():.1f} MB")

# Force prototype mode (skip CUDA/mamba imports)
os.environ['GROUNDTHINK_FORCE_PROTOTYPE'] = '1'

# Import only RWKV6 prototype directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ops.rwkv6_prototype import RWKV6Attention_Prototype
print(f"[Stage 2] After RWKV6 import: {get_memory_mb():.1f} MB")

# Minimal model (no full model.py import)
class TinyRWKV6(nn.Module):
    def __init__(self, vocab_size=1000, hidden=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.blocks = nn.ModuleList([
            RWKV6Attention_Prototype(hidden, num_heads=2, head_size=32)
            for _ in range(layers)
        ])
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Tie weights
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = x + block(x)[0]
        return self.head(x)

print(f"[Stage 3] Model class defined: {get_memory_mb():.1f} MB")

# Create tiny model
model = TinyRWKV6(vocab_size=1000, hidden=64, layers=2)
param_count = sum(p.numel() for p in model.parameters())
print(f"[Stage 4] Model created ({param_count/1e3:.1f}K params): {get_memory_mb():.1f} MB")

# Minimal training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
print(f"[Stage 5] Optimizer created: {get_memory_mb():.1f} MB")

# Synthetic data (no file loading)
batch_size = 1
seq_len = 32
print(f"\nTraining config: batch={batch_size}, seq={seq_len}, steps=5")

# Training loop
model.train()
for step in range(5):
    # Random input (no dataset loading)
    x = torch.randint(0, 1000, (batch_size, seq_len))
    y = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward
    logits = model(x)
    loss = criterion(logits.view(-1, 1000), y.view(-1))
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Step {step+1}/5: loss={loss.item():.4f}, mem={get_memory_mb():.1f} MB")
    
    # Aggressive cleanup
    del logits, loss
    gc.collect()

print(f"\n✓ SUCCESS: Training completed without crash")
print(f"[Final] Peak memory: {get_memory_mb():.1f} MB")
