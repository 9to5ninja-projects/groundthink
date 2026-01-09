"""
Small GroundThink - Train locally with real text.
~10M params, runs on 8GB VRAM.
Uses TinyShakespeare dataset.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path

from layers import GroundThinkBlock, RMSNorm

class SmallGroundThink(nn.Module):
    def __init__(self, vocab_size=32000, dim=256, n_layers=6, n_heads=8, head_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            GroundThinkBlock(dim=dim, n_heads=n_heads, head_dim=head_dim)
            for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        # Fix initialization
        self._init_weights()
    
    def _init_weights(self):
        # Scale embeddings down so logits don't explode
        # logit_std ≈ embed_std * sqrt(dim), so scale by 1/sqrt(dim)
        nn.init.normal_(self.embed.weight, std=0.02)
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_out(x)
        # Manual matmul for tied weights
        return x @ self.embed.weight.T

class TextDataset(Dataset):
    """Simple dataset from text file."""
    def __init__(self, text, tokenizer, seq_len=256):
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
        self.n_samples = len(self.tokens) // seq_len
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start + self.seq_len + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

class CharTokenizer:
    """Simple character-level tokenizer."""
    def __init__(self, text):
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)
    
    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)

def get_shakespeare():
    """Download or load Shakespeare text."""
    path = Path('shakespeare.txt')
    if not path.exists():
        print("Downloading Shakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    return path.read_text(encoding='utf-8')

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Config - ~10M params
    dim = 256
    n_layers = 6
    n_heads = 8
    head_dim = 32
    seq_len = 256
    batch_size = 8
    num_steps = 1000  # More steps
    lr = 1e-3  # Higher LR for faster convergence test
    
    # Print expected random loss
    print(f"Expected random loss: {torch.log(torch.tensor(65.0)).item():.2f}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    print("Loading Shakespeare...")
    text = get_shakespeare()
    print(f"Text length: {len(text):,} chars")
    
    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    dataset = TextDataset(text, tokenizer, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Samples: {len(dataset):,}")
    
    # Model
    model = SmallGroundThink(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim
    ).to(device)
    print(f"Parameters: {count_params(model):,}")
    
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
        for inputs, targets in loader:
            if step >= num_steps:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % 50 == 0:
                avg_loss = total_loss / 50
                elapsed = time.time() - start
                tok_per_sec = (step * batch_size * seq_len) / elapsed
                print(f"Step {step:4d} | Loss: {avg_loss:.4f} | {tok_per_sec:.0f} tok/s")
                total_loss = 0
    
    elapsed = time.time() - start
    print(f"\nDone! {elapsed:.1f}s, {step * batch_size * seq_len / elapsed:.0f} tok/s")
    
    # Generate sample
    print("\n--- Generation Sample (temp=0.8) ---")
    model.eval()
    import warnings
    warnings.filterwarnings("ignore", message="Input tensor shape suggests potential format mismatch")
    
    with torch.no_grad():
        prompt_text = "ROMEO:"
        prompt = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
        
        # Pad to at least 16 tokens for efficient processing
        min_len = 16
        if prompt.shape[1] < min_len:
            pad = torch.zeros(1, min_len - prompt.shape[1], dtype=torch.long, device=device)
            prompt_padded = torch.cat([pad, prompt], dim=1)
        else:
            prompt_padded = prompt
        
        for _ in range(300):
            logits = model(prompt_padded)
            # Only use the logit for the last real token
            probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1)
            prompt_padded = torch.cat([prompt_padded, next_token], dim=1)
        
        # Remove padding from output
        generated = tokenizer.decode(prompt_padded[0, min_len - prompt.shape[1]:].tolist())
        print(generated)
    
    # Also show greedy generation
    print("\n--- Generation Sample (greedy) ---")
    with torch.no_grad():
        prompt_text = "JULIET:"
        prompt = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
        
        if prompt.shape[1] < min_len:
            pad = torch.zeros(1, min_len - prompt.shape[1], dtype=torch.long, device=device)
            prompt_padded = torch.cat([pad, prompt], dim=1)
        else:
            prompt_padded = prompt
        
        for _ in range(200):
            logits = model(prompt_padded)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            prompt_padded = torch.cat([prompt_padded, next_token], dim=1)
        
        generated = tokenizer.decode(prompt_padded[0, min_len - prompt.shape[1]:].tolist())
        print(generated)
    
    # Save model
    torch.save(model.state_dict(), 'groundthink_small.pt')
    print("\n✅ Saved to groundthink_small.pt")

if __name__ == '__main__':
    main()
