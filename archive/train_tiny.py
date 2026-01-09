"""
Tiny GroundThink - Train locally on consumer GPU.
~5M params, runs on 6GB VRAM.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from layers import GroundThinkBlock, RMSNorm

class TinyGroundThink(nn.Module):
    def __init__(self, vocab_size=10000, dim=128, n_layers=4, n_heads=4, head_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            GroundThinkBlock(dim=dim, n_heads=n_heads, head_dim=head_dim)
            for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Tie embeddings
        self.head.weight = self.embed.weight
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_out(x)
        return self.head(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Config
    vocab_size = 10000
    dim = 128
    n_layers = 4
    n_heads = 4
    head_dim = 32
    seq_len = 128
    batch_size = 8
    num_samples = 1000
    num_steps = 100
    lr = 3e-4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Model
    model = TinyGroundThink(vocab_size, dim, n_layers, n_heads, head_dim).to(device)
    print(f"Parameters: {count_params(model):,}")
    
    # Fake data (random tokens)
    print("Creating synthetic data...")
    data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))
    inputs = data[:, :-1]
    targets = data[:, 1:]
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"\nTraining for {num_steps} steps...")
    model.train()
    step = 0
    total_loss = 0
    start = time.time()
    
    while step < num_steps:
        for batch_inputs, batch_targets in loader:
            if step >= num_steps:
                break
            
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits.view(-1, vocab_size), batch_targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % 10 == 0:
                avg_loss = total_loss / 10
                elapsed = time.time() - start
                tok_per_sec = (step * batch_size * seq_len) / elapsed
                print(f"Step {step:4d} | Loss: {avg_loss:.4f} | {tok_per_sec:.0f} tok/s")
                total_loss = 0
    
    elapsed = time.time() - start
    print(f"\nDone! {elapsed:.1f}s total, {step * batch_size * seq_len / elapsed:.0f} tok/s avg")
    
    # Quick generation test
    print("\nGeneration test:")
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, vocab_size, (1, 10)).to(device)
        for _ in range(20):
            logits = model(prompt)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            prompt = torch.cat([prompt, next_token], dim=1)
        print(f"Generated {prompt.shape[1]} tokens")
    
    print("\n✅ Training complete!")

if __name__ == '__main__':
    main()
