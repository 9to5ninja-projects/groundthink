#!/usr/bin/env python3
"""
Task 62: GPT-2 vs GF-MH on WikiText-103 (10MB subset)
Uses BPE tokenization as required by V5_GATING.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import time
import gc

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load tokenizer
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer_wikitext.json")
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f"Vocab: {VOCAB_SIZE}")

# Load SMALL subset of WikiText-103 (first 10MB only)
print("Loading 10MB subset...")
with open("data/wikitext103/train.txt", "r") as f:
    text = f.read(10_000_000)  # 10MB only

# Tokenize
encoded = tokenizer.encode(text)
data = torch.tensor(encoded.ids, dtype=torch.long)
print(f"Tokens: {len(data):,}")

# Split
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]
del data, text, encoded
gc.collect()

# Config
BATCH_SIZE = 16
SEQ_LEN = 64
LR = 3e-4
STEPS = 300

def get_batch(data, batch_size=BATCH_SIZE):
    idx = torch.randint(0, len(data) - SEQ_LEN - 1, (batch_size,))
    x = torch.stack([data[i:i+SEQ_LEN] for i in idx])
    y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in idx])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate(model, data, is_gpt2=False):
    model.eval()
    losses = []
    for _ in range(20):
        x, y = get_batch(data)
        if is_gpt2:
            _, loss = model(x, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

def train(model, name, is_gpt2=False):
    print(f"\n{'='*50}")
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"{'='*50}")
    
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    model.train()
    
    for step in range(STEPS):
        x, y = get_batch(train_data)
        opt.zero_grad()
        
        if is_gpt2:
            _, loss = model(x, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if (step + 1) % 100 == 0:
            val = evaluate(model, val_data, is_gpt2)
            print(f"  Step {step+1}: train={loss.item():.3f} val={val:.3f}")
    
    train_time = time.time() - start
    val_loss = evaluate(model, val_data, is_gpt2)
    mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
    
    return {"val_loss": val_loss, "time": train_time, "mem": mem}

# Run comparison
from models import get_model

print("\n" + "="*50)
print("Task 62: GPT-2 vs GF-MH (WikiText-103 + BPE)")
print("="*50)

# GPT-2
torch.manual_seed(SEED)
gpt2 = get_model("GPT2", vocab_size=VOCAB_SIZE)
r1 = train(gpt2, "GPT-2", is_gpt2=True)
del gpt2
torch.cuda.empty_cache()
gc.collect()

# GF-MH
torch.manual_seed(SEED)
gfmh = get_model("GF-MH", vocab_size=VOCAB_SIZE)
r2 = train(gfmh, "GF-MH", is_gpt2=False)
del gfmh

# Results
print("\n" + "="*50)
print("RESULTS")
print("="*50)
ratio = r2["val_loss"] / r1["val_loss"]
print(f"GPT-2:  loss={r1['val_loss']:.3f}  time={r1['time']:.1f}s  mem={r1['mem']:.0f}MB")
print(f"GF-MH:  loss={r2['val_loss']:.3f}  time={r2['time']:.1f}s  mem={r2['mem']:.0f}MB")
print(f"Ratio:  {ratio:.3f}")

if ratio <= 0.95:
    print("Verdict: EXCELLENT - GF-MH significantly better")
elif ratio <= 1.05:
    print("Verdict: EQUIVALENT")
elif ratio <= 1.20:
    print("Verdict: ACCEPTABLE - GF-MH slightly worse")
else:
    print("Verdict: FAIL - GF-MH significantly worse")
