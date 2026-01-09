"""
Train GroundThink with proper hybrid SSM training recipe.

Key fixes from research:
1. Per-component learning rates (RWKV decay params vs Mamba-like params)
2. Proper initialization for w (decay) and Delta projections
3. State norm monitoring
4. Longer warmup for SSM stability
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path
import math

from layers import GroundThinkBlock, RMSNorm


class GroundThinkModel(nn.Module):
    def __init__(self, vocab_size=256, dim=256, n_layers=6, n_heads=8, head_dim=32):
        super().__init__()
        self.dim = dim
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
        """Proper hybrid initialization"""
        # Embedding: small to prevent logit explosion
        nn.init.normal_(self.embed.weight, std=0.02)
        
        for block in self.blocks:
            tm = block.time_mixing
            
            # RWKV-style decay (time_decay): initialize to produce moderate decay
            # exp(-exp(w)) where w ~ 0 gives decay ~ 0.37
            # We want decay ~ 0.9 (slow forgetting), so init to negative values
            nn.init.uniform_(tm.time_decay.weight, -1.0, -0.5)
            
            # Key/Value: small init
            nn.init.normal_(tm.key.weight, std=0.02)
            nn.init.normal_(tm.value.weight, std=0.02)
            
            # Receptance: slightly positive (read gate should be open initially)
            nn.init.normal_(tm.receptance.weight, std=0.02)
            
            # Gate: initialize to ~0.5 sigmoid output (weight near 0)
            nn.init.normal_(tm.gate.weight, std=0.01)
            
            # Output projection: small to not disrupt residual
            nn.init.normal_(tm.out_proj.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
            
            # Grounding mechanism
            if tm.use_grounding:
                # Base decay: init to small positive (slow decay floor)
                nn.init.constant_(tm.grounding.base_decay, 0.5)
                # Residual weight: start at 1
                nn.init.constant_(tm.grounding.res_weight, 1.0)
    
    def get_param_groups(self, lr_base=3e-4, lr_decay=1e-3, wd=0.1):
        """
        Split parameters into groups with different learning rates.
        
        - Decay params (time_decay, grounding): higher LR, no weight decay
        - Other params: base LR, with weight decay
        """
        decay_params = []
        nodecay_params = []
        regular_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'time_decay' in name or 'base_decay' in name or 'grounding' in name:
                # Decay-related params: higher LR, no WD
                decay_params.append(param)
            elif param.dim() < 2 or 'ln' in name or 'norm' in name or 'embed' in name:
                # 1D params, norms, embeddings: no weight decay
                nodecay_params.append(param)
            else:
                # Regular params: base LR, with WD
                regular_params.append(param)
        
        return [
            {'params': decay_params, 'lr': lr_decay, 'weight_decay': 0.0, 'name': 'decay'},
            {'params': nodecay_params, 'lr': lr_base, 'weight_decay': 0.0, 'name': 'nodecay'},
            {'params': regular_params, 'lr': lr_base, 'weight_decay': wd, 'name': 'regular'},
        ]
    
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
        
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        else:
            tokens = tokens[:self.seq_len + 1]
        
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
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


def compute_state_norms(model, sample_input):
    """Monitor state norms to detect explosion/collapse"""
    model.eval()
    with torch.no_grad():
        x = model.embed(sample_input)
        norms = []
        for i, block in enumerate(model.blocks):
            x, state = block(x)
            if state is not None:
                norm = state.norm().item()
                norms.append(norm)
    model.train()
    return norms


def main():
    # Config
    dim = 256
    n_layers = 6
    n_heads = 8
    head_dim = 32
    seq_len = 256
    batch_size = 16
    num_steps = 5000
    lr_base = 3e-4
    lr_decay = 1e-3  # Higher LR for decay params
    warmup_steps = 500  # Longer warmup for SSM stability
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data - use clean filtered version if available
    data_path = Path('data/mixed_training_data_clean.txt')
    if not data_path.exists():
        data_path = Path('data/mixed_training_data.txt')
    if not data_path.exists():
        print("ERROR: Run download_training_data.py first!")
        return
    
    print("Building tokenizer...")
    with open(data_path, 'r', encoding='utf-8') as f:
        sample_text = f.read(1_000_000)
    tokenizer = CharTokenizer(sample_text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    expected_loss = math.log(tokenizer.vocab_size)
    print(f"Expected random loss: {expected_loss:.2f}")
    
    dataset = LineDataset(data_path, tokenizer, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Model with proper init
    model = GroundThinkModel(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Per-component optimizer
    param_groups = model.get_param_groups(lr_base=lr_base, lr_decay=lr_decay, wd=0.1)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    
    # Print param group sizes
    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: {n:,} params, lr={pg['lr']:.1e}, wd={pg['weight_decay']}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # LR scheduler: linear warmup then cosine
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (num_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Get sample for state monitoring
    sample_batch = next(iter(loader))[0][:1].to(device)
    
    # Initial state norms
    init_norms = compute_state_norms(model, sample_batch)
    print(f"Initial state norms: {[f'{n:.2f}' for n in init_norms]}")
    
    # Train
    print(f"\nTraining for {num_steps} steps (warmup: {warmup_steps})...")
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
            
            # Gradient clipping before backward for stability
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % 100 == 0:
                avg_loss = total_loss / 100
                elapsed = time.time() - start
                tok_per_sec = (step * batch_size * seq_len) / elapsed
                current_lr = scheduler.get_last_lr()[0]
                
                # State health check every 500 steps
                state_info = ""
                if step % 500 == 0:
                    norms = compute_state_norms(model, sample_batch)
                    state_info = f" | states: {[f'{n:.1f}' for n in norms[:3]]}"
                
                print(f"Step {step:4d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                      f"Grad: {grad_norm:.2f} | {tok_per_sec:.0f} tok/s{state_info}")
                
                total_loss = 0
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
    
    elapsed = time.time() - start
    print(f"\nDone! {elapsed:.1f}s, {step * batch_size * seq_len / elapsed:.0f} tok/s")
    print(f"Best loss: {best_loss:.4f}")
    
    # Final state norms
    final_norms = compute_state_norms(model, sample_batch)
    print(f"Final state norms: {[f'{n:.2f}' for n in final_norms]}")
    
    # Generate samples
    print("\n--- Generation Samples ---")
    model.eval()
    
    prompts = [
        "Once upon a time",
        "The old man looked",
        "She walked into the room",
        "I think that",
        "He said to her",
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
            if '\n' in generated:
                generated = generated[:generated.index('\n')]
            print(f"  → {generated[:200]}")
    
    # Save
    torch.save({
        'model': model.state_dict(),
        'vocab': tokenizer.char_to_id,
        'config': {'dim': dim, 'n_layers': n_layers, 'n_heads': n_heads, 'head_dim': head_dim}
    }, 'groundthink_hybrid_v2.pt')
    print("\n✅ Saved to groundthink_hybrid_v2.pt")


if __name__ == '__main__':
    main()
