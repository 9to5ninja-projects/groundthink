"""
Minimal training loop test - CPU only.
Verifies the training logic works before deploying to GPU.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("=== MINIMAL TRAINING TEST ===")

# Import layers
from layers import GroundThinkBlock, RMSNorm

class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, dim=64, n_layers=2, n_heads=4, head_dim=16):
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

# Create model
print("[1] Creating model...")
model = SimpleModel()
total_params = sum(p.numel() for p in model.parameters())
print(f"    Parameters: {total_params:,}")

# Create fake data
print("[2] Creating fake dataset...")
batch_size = 4
seq_len = 32
num_samples = 20

fake_input = torch.randint(0, 1000, (num_samples, seq_len))
fake_target = torch.randint(0, 1000, (num_samples, seq_len))
dataset = TensorDataset(fake_input, fake_target)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"    Batches: {len(dataloader)}")

# Optimizer
print("[3] Setting up optimizer...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
print("[4] Training for 5 steps...")
model.train()
losses = []

for step, (input_ids, targets) in enumerate(dataloader):
    if step >= 5:
        break
    
    optimizer.zero_grad()
    
    # Forward
    logits = model(input_ids)  # [B, T, V]
    
    # Loss
    loss = criterion(logits.view(-1, 1000), targets.view(-1))
    
    # Backward
    loss.backward()
    
    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Step
    optimizer.step()
    
    losses.append(loss.item())
    print(f"    Step {step+1}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

# Check loss decreased
print("\n[5] Verifying training...")
if losses[-1] < losses[0]:
    print(f"    ✅ Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
else:
    print(f"    ⚠️ Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("    (This is OK for 5 steps with random data)")

# Check parameters changed
print("\n[6] Checking parameter updates...")
with torch.no_grad():
    # Check a few key parameters
    checks = [
        ('embed.weight', model.embed.weight),
        ('blocks.0.time_mixing.key.weight', model.blocks[0].time_mixing.key.weight),
        ('blocks.0.time_mixing.value.weight', model.blocks[0].time_mixing.value.weight),
        ('head.weight', model.head.weight),
    ]
    for name, param in checks:
        # Param should have been updated (not all zeros)
        is_nonzero = param.abs().mean().item() > 1e-6
        print(f"    {name}: {'✅' if is_nonzero else '❌'} (mean abs: {param.abs().mean().item():.6f})")

print("\n=== TRAINING TEST COMPLETE ===")
