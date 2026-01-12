"""
Minimal Pure-PyTorch Training Test (NO FLA, NO mamba_ssm)

Uses grounded_layers.py which is 100% PyTorch.
Purpose: Establish baseline memory footprint for WSL development.

Memory target: <600 MB peak
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import resource
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"[0] Start: {mem():.0f} MB")

import torch
import torch.nn as nn
print(f"[1] PyTorch: {mem():.0f} MB")

# Import pure PyTorch layers (NO FLA, NO mamba_ssm)
from grounded_layers import GroundedMambaBlock
print(f"[2] Layers: {mem():.0f} MB")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TinyGroundedModel(nn.Module):
    """Minimal model using pure PyTorch SSM"""
    def __init__(self, vocab_size=10000, dim=128, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            GroundedMambaBlock(dim=dim, ssm_dim=16, num_heads=4)
            for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Tie weights
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x, _ = block(x)  # Block returns (output, states)
        x = self.ln_out(x)
        return self.head(x)


def main():
    # Ultra-conservative config for WSL
    vocab_size = 10000
    dim = 128
    n_layers = 4
    seq_len = 64
    batch_size = 1
    num_steps = 20
    
    device = 'cpu'  # Stay on CPU for memory safety
    
    # Create model
    model = TinyGroundedModel(vocab_size, dim, n_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[3] Model ({n_params/1e6:.2f}M params): {mem():.0f} MB")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    print(f"[4] Optimizer: {mem():.0f} MB")
    
    # Training loop with synthetic data
    print(f"\nTraining: batch={batch_size}, seq={seq_len}, steps={num_steps}")
    model.train()
    
    for step in range(num_steps):
        # Synthetic data
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}/{num_steps}: loss={loss.item():.4f}, mem={mem():.0f} MB")
    
    print(f"\n✓ SUCCESS: {mem():.0f} MB peak")
    print("Pure PyTorch SSM training works on WSL!")


if __name__ == '__main__':
    main()
