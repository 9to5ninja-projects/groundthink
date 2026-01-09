"""
Train GroundThink on mixed dataset (TinyStories + Books + Dialogue)
5M params, local training
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
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_out(x)
        return x @ self.embed.weight.T

class LineDataset(Dataset):
    """Dataset from text file with one sample per line."""
    def __init__(self, filepath, tokenizer, seq_len=256):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        print(f"Loading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if len(line.strip()) > 50]
        print(f"  Loaded {len(self.lines):,} samples")
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        text = self.lines[idx]
        tokens = self.tokenizer.encode(text)
        
        # Pad or truncate to seq_len + 1
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        else:
            tokens = tokens[:self.seq_len + 1]
        
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

class CharTokenizer:
    """Character-level tokenizer."""
    def __init__(self, text):
        chars = sorted(set(text))
        # Ensure space and common chars are included
        for c in ' \n\t.,!?\'"-:;()':
            if c not in chars:
                chars.append(c)
        chars = sorted(set(chars))
        
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)
    
    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Config - 5M params
    dim = 256
    n_layers = 6
    n_heads = 8
    head_dim = 32
    seq_len = 256
    batch_size = 16
    num_steps = 3000
    lr = 3e-4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    data_path = Path('data/mixed_training_data.txt')
    if not data_path.exists():
        print("ERROR: Run download_training_data.py first!")
        return
    
    # Build tokenizer from data
    print("Building tokenizer...")
    with open(data_path, 'r', encoding='utf-8') as f:
        sample_text = f.read(1_000_000)  # First 1M chars for vocab
    tokenizer = CharTokenizer(sample_text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Expected random loss
    expected_loss = torch.log(torch.tensor(float(tokenizer.vocab_size))).item()
    print(f"Expected random loss: {expected_loss:.2f}")
    
    # Dataset
    dataset = LineDataset(data_path, tokenizer, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = SmallGroundThink(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim
    ).to(device)
    print(f"Parameters: {count_params(model):,}")
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # LR scheduler - warmup then cosine
    def lr_lambda(step):
        warmup = 100
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + torch.cos(torch.tensor((step - warmup) / (num_steps - warmup) * 3.14159)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Train
    print(f"\nTraining for {num_steps} steps...")
    model.train()
    step = 0
    total_loss = 0
    start = time.time()
    best_loss = float('inf')
    
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
            scheduler.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % 100 == 0:
                avg_loss = total_loss / 100
                elapsed = time.time() - start
                tok_per_sec = (step * batch_size * seq_len) / elapsed
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {step:4d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | {tok_per_sec:.0f} tok/s")
                total_loss = 0
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
    
    elapsed = time.time() - start
    print(f"\nDone! {elapsed:.1f}s, {step * batch_size * seq_len / elapsed:.0f} tok/s")
    print(f"Best loss: {best_loss:.4f}")
    
    # Generate samples
    print("\n--- Generation: Story Start ---")
    model.eval()
    import warnings
    warnings.filterwarnings("ignore")
    
    prompts = [
        "Once upon a time",
        "The old man looked",
        "She walked into the room",
        "I think that",
    ]
    
    for prompt_text in prompts:
        print(f"\nPrompt: '{prompt_text}'")
        with torch.no_grad():
            prompt = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
            
            min_len = 16
            if prompt.shape[1] < min_len:
                pad = torch.zeros(1, min_len - prompt.shape[1], dtype=torch.long, device=device)
                prompt_padded = torch.cat([pad, prompt], dim=1)
            else:
                prompt_padded = prompt
            
            for _ in range(150):
                logits = model(prompt_padded)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)
                prompt_padded = torch.cat([prompt_padded, next_token], dim=1)
            
            generated = tokenizer.decode(prompt_padded[0, min_len - prompt.shape[1]:].tolist())
            # Truncate at first newline or after 200 chars
            if '\n' in generated:
                generated = generated[:generated.index('\n')]
            print(f"  → {generated[:200]}")
    
    # Save
    torch.save({
        'model': model.state_dict(),
        'vocab': tokenizer.char_to_id,
        'config': {'dim': dim, 'n_layers': n_layers, 'n_heads': n_heads, 'head_dim': head_dim}
    }, 'groundthink_mixed.pt')
    print("\n✅ Saved to groundthink_mixed.pt")

if __name__ == '__main__':
    main()
