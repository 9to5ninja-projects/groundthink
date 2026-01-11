#!/usr/bin/env python3
"""
Task 62: GPT-2 vs GF-MH Comparison on WikiText-103

Per V5_GATING.md requirements:
- Same data (WikiText-103)
- Same tokenizer (BPE 16K)
- Same seed (42)
- Same batch order
- Same training config
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer

# Seed everything
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# === Config ===
VOCAB_SIZE = 16000
BATCH_SIZE = 32
SEQ_LEN = 128
LR = 3e-4
STEPS = 1000
EVAL_EVERY = 100

print("=" * 60)
print("Task 62: GPT-2 vs GF-MH on WikiText-103")
print("=" * 60)

# === Load tokenizer ===
print("\nLoading BPE tokenizer...")
tokenizer = Tokenizer.from_file("data/tokenizer_wikitext.json")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# === Load and tokenize data ===
print("\nLoading WikiText-103...")
train_text = Path("data/wikitext103/train.txt").read_text()
val_text = Path("data/wikitext103/valid.txt").read_text()

print("Tokenizing (this may take a minute)...")
train_ids = tokenizer.encode(train_text).ids
val_ids = tokenizer.encode(val_text).ids

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data = torch.tensor(val_ids, dtype=torch.long)

print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")

# === Create batch order ===
n_batches = len(train_data) // (BATCH_SIZE * SEQ_LEN)
batch_order = np.random.permutation(n_batches)
np.save("data/batch_order_wikitext.npy", batch_order)
print(f"Batch order saved: {n_batches} batches")

def get_batch(data, batch_idx):
    start = batch_idx * BATCH_SIZE * SEQ_LEN
    x = data[start:start + BATCH_SIZE * SEQ_LEN].view(BATCH_SIZE, SEQ_LEN)
    y = data[start + 1:start + BATCH_SIZE * SEQ_LEN + 1].view(BATCH_SIZE, SEQ_LEN)
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate(model, data, n_batches=20, is_gpt2=False):
    model.eval()
    losses = []
    for i in range(min(n_batches, len(data) // (BATCH_SIZE * SEQ_LEN))):
        x, y = get_batch(data, i)
        if is_gpt2:
            logits, loss = model(x, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

def train_model(model, name, steps=STEPS, is_gpt2=False):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    losses = []
    val_losses = []
    start_time = time.time()
    
    model.train()
    for step in range(steps):
        batch_idx = batch_order[step % len(batch_order)]
        x, y = get_batch(train_data, batch_idx)
        
        optimizer.zero_grad()
        if is_gpt2:
            logits, loss = model(x, y)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_data, is_gpt2=is_gpt2)
            val_losses.append(val_loss)
            elapsed = time.time() - start_time
            print(f"  Step {step+1:4d} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
    
    return {
        "name": name,
        "params": params,
        "final_train_loss": np.mean(losses[-50:]),
        "final_val_loss": val_losses[-1] if val_losses else evaluate(model, val_data, is_gpt2=is_gpt2),
        "train_time": total_time,
        "peak_memory_mb": peak_mem,
        "tokens_per_sec": (steps * BATCH_SIZE * SEQ_LEN) / total_time,
    }

# === Import models ===
from models import get_model

results = {}

# === Train GPT-2 ===
torch.manual_seed(SEED)
np.random.seed(SEED)
gpt2 = get_model("GPT2", vocab_size=VOCAB_SIZE)
results["GPT2"] = train_model(gpt2, "GPT-2 (Transformer)", is_gpt2=True)
del gpt2
if device.type == "cuda":
    torch.cuda.empty_cache()

# === Train GF-MH ===
torch.manual_seed(SEED)
np.random.seed(SEED)
gfmh = get_model("GF-MH", vocab_size=VOCAB_SIZE)
results["GF-MH"] = train_model(gfmh, "GF-MH (Hybrid)", is_gpt2=False)
del gfmh

# === Results ===
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

gpt2_r = results["GPT2"]
gfmh_r = results["GF-MH"]

print(f"\n{'Metric':<25} {'GPT-2':<15} {'GF-MH':<15} {'Ratio':<10}")
print("-" * 65)

loss_ratio = gfmh_r["final_val_loss"] / gpt2_r["final_val_loss"]
time_ratio = gfmh_r["train_time"] / gpt2_r["train_time"]
speed_ratio = gfmh_r["tokens_per_sec"] / gpt2_r["tokens_per_sec"]
mem_ratio = gfmh_r["peak_memory_mb"] / gpt2_r["peak_memory_mb"] if gpt2_r["peak_memory_mb"] > 0 else 1.0

print(f"{'Parameters':<25} {gpt2_r['params']:,} {gfmh_r['params']:,}")
print(f"{'Final Val Loss':<25} {gpt2_r['final_val_loss']:<15.4f} {gfmh_r['final_val_loss']:<15.4f} {loss_ratio:<10.3f}")
print(f"{'Training Time (s)':<25} {gpt2_r['train_time']:<15.1f} {gfmh_r['train_time']:<15.1f} {time_ratio:<10.2f}x")
print(f"{'Tokens/sec':<25} {gpt2_r['tokens_per_sec']:<15.0f} {gfmh_r['tokens_per_sec']:<15.0f} {speed_ratio:<10.2f}x")
print(f"{'Peak Memory (MB)':<25} {gpt2_r['peak_memory_mb']:<15.1f} {gfmh_r['peak_memory_mb']:<15.1f} {mem_ratio:<10.2f}x")

# === Verdict per V5_GATING.md thresholds ===
print("\n" + "=" * 60)
print("VERDICT (per V5_GATING.md thresholds)")
print("=" * 60)

if loss_ratio <= 0.95:
    loss_verdict = "🏆 EXCELLENT: GF-MH significantly better"
elif loss_ratio <= 1.05:
    loss_verdict = "✓ EQUIVALENT: Within 5%"
elif loss_ratio <= 1.15:
    loss_verdict = "⚠ GOOD ENOUGH: Within 15%"
elif loss_ratio <= 1.20:
    loss_verdict = "⚠ MARGINAL: Within 20%"
else:
    loss_verdict = "✗ FAIL: >20% worse, stop and debug"

print(f"\nLoss Ratio: {loss_ratio:.3f}")
print(f"Loss Verdict: {loss_verdict}")

# Speed assessment
if speed_ratio >= 1.5:
    speed_verdict = "✓ 50%+ faster"
elif speed_ratio >= 1.2:
    speed_verdict = "✓ 20%+ faster"
elif speed_ratio >= 0.8:
    speed_verdict = "~ Similar speed"
else:
    speed_verdict = "⚠ Slower"

print(f"Speed Verdict: {speed_verdict}")

# Overall
print("\n" + "-" * 60)
if loss_ratio <= 1.20:
    print("PROCEED TO 8M: Loss within acceptable range")
else:
    print("STOP: Fix architecture before scaling")

print("=" * 60)
