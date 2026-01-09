"""
Debug: check model outputs to understand the high initial loss.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from layers import GroundThinkBlock, RMSNorm

class SmallGroundThink(nn.Module):
    def __init__(self, vocab_size=65, dim=256, n_layers=6, n_heads=8, head_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            GroundThinkBlock(dim=dim, n_heads=n_heads, head_dim=head_dim)
            for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(dim)
        self.head = nn.Linear(vocab_size, dim, bias=False)  # Note: transposed
        self.head.weight = self.embed.weight
        
        # Fix initialization
        nn.init.normal_(self.embed.weight, std=0.02)
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_out(x)
        # Manual matmul since weight is [vocab, dim]
        return x @ self.embed.weight.T

# Create model
model = SmallGroundThink()
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Random input
x = torch.randint(0, 65, (2, 32))

# Forward
with torch.no_grad():
    logits = model(x)
    
print(f"Logits shape: {logits.shape}")
print(f"Logits mean: {logits.mean().item():.4f}")
print(f"Logits std: {logits.std().item():.4f}")
print(f"Logits min: {logits.min().item():.4f}")
print(f"Logits max: {logits.max().item():.4f}")

# Check what loss we'd get
targets = torch.randint(0, 65, (2, 32))
loss = nn.CrossEntropyLoss()(logits.view(-1, 65), targets.view(-1))
print(f"\nRandom loss: {loss.item():.4f}")
print(f"Expected (ln(65)): {torch.log(torch.tensor(65.0)).item():.4f}")

# Check the distribution of predictions
probs = torch.softmax(logits[0, 0], dim=-1)
print(f"\nFirst token probs - min: {probs.min():.6f}, max: {probs.max():.6f}")
print(f"Entropy: {(-probs * probs.log()).sum().item():.4f}")
print(f"Max entropy (uniform): {torch.log(torch.tensor(65.0)).item():.4f}")

# Check embeddings
print(f"\nEmbedding weight std: {model.embed.weight.std().item():.4f}")
print(f"Head weight std: {model.head.weight.std().item():.4f}")

# Check intermediate activations
with torch.no_grad():
    emb = model.embed(x)
    print(f"\nAfter embedding - mean: {emb.mean():.4f}, std: {emb.std():.4f}")
    
    out = emb
    for i, block in enumerate(model.blocks):
        out, _ = block(out)
        print(f"After block {i} - mean: {out.mean():.4f}, std: {out.std():.4f}")
    
    out = model.ln_out(out)
    print(f"After ln_out - mean: {out.mean():.4f}, std: {out.std():.4f}")
