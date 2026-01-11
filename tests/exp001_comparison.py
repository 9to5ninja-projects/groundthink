#!/usr/bin/env python3
"""
EXP-001: GPT-2 vs GF-MH Controlled Comparison

Scientific method:
- Same data, same seed, same batch order
- Both models ~5.6M params
- Compare: loss, speed, memory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Seed everything
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

from models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("=" * 60)

# === Config (control variables) ===
VOCAB_SIZE = 10000
BATCH_SIZE = 32
SEQ_LEN = 64
LR = 3e-4
STEPS = 200
EVAL_EVERY = 50

# === Load data ===
data_path = Path("data/shakespeare.txt")
if data_path.exists():
    text = data_path.read_text()
    # Simple char-level tokenization (map to vocab range)
    chars = sorted(set(text))
    char_to_idx = {c: i % VOCAB_SIZE for i, c in enumerate(chars)}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    print(f"Data: {len(data):,} tokens from shakespeare.txt")
else:
    # Synthetic data for testing
    data = torch.randint(0, VOCAB_SIZE, (100000,))
    print(f"Data: synthetic {len(data):,} tokens")

# Split train/val
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]
print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")

# Save batch order for reproducibility
n_batches = len(train_data) // (BATCH_SIZE * SEQ_LEN)
batch_order = np.random.permutation(n_batches)
np.save("data/batch_order.npy", batch_order)
print(f"Batch order saved: {n_batches} batches")

def get_batch(split_data, batch_idx):
    """Get a batch at specific index."""
    start = batch_idx * BATCH_SIZE * SEQ_LEN
    x = split_data[start:start + BATCH_SIZE * SEQ_LEN].view(BATCH_SIZE, SEQ_LEN)
    y = split_data[start + 1:start + BATCH_SIZE * SEQ_LEN + 1].view(BATCH_SIZE, SEQ_LEN)
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate(model, data, n_batches=10, is_gpt2=False):
    """Evaluate on validation data."""
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
    """Train model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    
    # Reset CUDA memory tracking
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
        "final_train_loss": np.mean(losses[-20:]),
        "final_val_loss": val_losses[-1] if val_losses else evaluate(model, val_data, is_gpt2=is_gpt2),
        "train_time": total_time,
        "peak_memory_mb": peak_mem,
        "tokens_per_sec": (steps * BATCH_SIZE * SEQ_LEN) / total_time,
    }

# === Run comparison ===
print("\n" + "=" * 60)
print("EXP-001: GPT-2 vs GF-MH Comparison")
print("=" * 60)

# Reset seed before each model
results = {}

# Train GPT-2
torch.manual_seed(SEED)
gpt2 = get_model("GPT2", vocab_size=VOCAB_SIZE)
results["GPT2"] = train_model(gpt2, "GPT-2 (Transformer)", is_gpt2=True)
del gpt2
torch.cuda.empty_cache() if device.type == "cuda" else None

# Train GF-MH
torch.manual_seed(SEED)
gfmh = get_model("GF-MH", vocab_size=VOCAB_SIZE)
results["GF-MH"] = train_model(gfmh, "GF-MH (Hybrid)", is_gpt2=False)
del gfmh

# === Analysis ===
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

print(f"{'Final Val Loss':<25} {gpt2_r['final_val_loss']:<15.4f} {gfmh_r['final_val_loss']:<15.4f} {loss_ratio:<10.3f}")
print(f"{'Training Time (s)':<25} {gpt2_r['train_time']:<15.1f} {gfmh_r['train_time']:<15.1f} {time_ratio:<10.2f}x")
print(f"{'Tokens/sec':<25} {gpt2_r['tokens_per_sec']:<15.0f} {gfmh_r['tokens_per_sec']:<15.0f} {speed_ratio:<10.2f}x")
print(f"{'Peak Memory (MB)':<25} {gpt2_r['peak_memory_mb']:<15.1f} {gfmh_r['peak_memory_mb']:<15.1f} {mem_ratio:<10.2f}x")

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

# Apply thresholds from protocol
if loss_ratio <= 0.95:
    verdict = "🏆 EXCELLENT: GF-MH significantly better"
elif loss_ratio <= 1.05:
    verdict = "✓ EQUIVALENT: Similar performance"
elif loss_ratio <= 1.20:
    verdict = "⚠ ACCEPTABLE: GF-MH slightly worse"
elif loss_ratio <= 1.30:
    verdict = "⚠ CONCERNING: GF-MH notably worse"
else:
    verdict = "✗ FAIL: GF-MH significantly worse"

print(f"\nLoss Ratio: {loss_ratio:.3f}")
print(f"Verdict: {verdict}")

# Secondary advantages
advantages = []
if speed_ratio > 1.0:
    advantages.append(f"Speed: {speed_ratio:.1f}x faster")
if mem_ratio < 1.0:
    advantages.append(f"Memory: {1/mem_ratio:.1f}x less")

if advantages:
    print(f"Advantages: {', '.join(advantages)}")

print("\n" + "=" * 60)
