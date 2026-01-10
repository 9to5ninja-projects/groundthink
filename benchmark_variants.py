"""
Benchmark All Fusion Variants - Survival of the Fittest

Quick benchmark to find the standout winner among fusion strategies.
All variants tested with identical:
- Dataset (Shakespeare)
- Hyperparameters (LR, batch, etc.)  
- Training steps (500 each)
- Evaluation criteria

Variants:
- HY: Hybrid per-channel gains (256 fusion params)
- WS: Weighted Sum alpha (1 fusion param)
- GF: Gated Fusion (257 fusion params)
- RF: Residual Fusion (16K fusion params)
- CP: Concatenate + Project (33K fusion params)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import math
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Import model registry
from models import get_model

from data_loader import load_stateful_dataset

# ============ Config ============
# Same config for ALL variants - fair comparison
CONFIG = {
    'lr': 3e-4,
    'warmup_steps': 50,
    'weight_decay': 0.1,
    'batch_size': 64,
    'seq_len': 64,
    'max_steps': 500,       # Quick benchmark
    'eval_every': 50,       # Eval every 50 steps
    'use_amp': True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ Variants ============
# Fusion variants (Task 14) - use registry names
FUSION_VARIANTS = {
    'HY': 'Hybrid Per-Channel',
    'WS': 'Weighted Sum',
    'GF': 'Gated Fusion',
    'RF': 'Residual Fusion',
    'CP': 'Concat+Project',
}

# Ratio variants (Task 15-16) - all use GF fusion
RATIO_VARIANTS = {
    'GF': 'GF Balanced',
    'GF-RH': 'GF RWKV-Heavy',
    'GF-MH': 'GF Mamba-Heavy',
}

# Default to ratio variants for Phase 2
VARIANTS = RATIO_VARIANTS


def get_lr(step, warmup_steps, max_steps, base_lr):
    """Cosine LR with warmup"""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.1 + base_lr * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def evaluate_val(model, dataset, n_batches=5):
    """Quick validation loss"""
    model.eval()
    dataset.reset_val()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i in range(n_batches):
            batch = dataset.get_val_batch()
            if batch is None:
                break
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda', enabled=CONFIG['use_amp']):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)


def train_variant(name, desc, dataset, vocab_size):
    """Train one variant and return metrics"""
    print(f"\n{'='*60}")
    print(f"VARIANT: {name} - {desc}")
    print(f"{'='*60}")
    
    # Create model using registry
    model = get_model(name, vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.95),
    )
    
    scaler = GradScaler() if CONFIG['use_amp'] else None
    
    # Training loop
    model.train()
    
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'steps': [],
        'time': 0,
        'tokens': 0,
        'n_params': n_params,
    }
    
    start_time = time.time()
    running_loss = 0.0
    n_loss = 0
    
    for step in range(1, CONFIG['max_steps'] + 1):
        # Get batch using dataset indexing
        x, y, _ = dataset[step % len(dataset)]
        x, y = x.to(device), y.to(device)
        
        # LR schedule
        lr = get_lr(step, CONFIG['warmup_steps'], CONFIG['max_steps'], CONFIG['lr'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward
        optimizer.zero_grad()
        
        if CONFIG['use_amp']:
            with autocast(device_type='cuda'):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        running_loss += loss.item()
        n_loss += 1
        metrics['tokens'] += x.numel()
        
        # Eval
        if step % CONFIG['eval_every'] == 0:
            avg_loss = running_loss / n_loss
            metrics['train_losses'].append(avg_loss)
            metrics['steps'].append(step)
            
            # Val loss
            val_loss = evaluate_val(model, dataset)
            metrics['val_losses'].append(val_loss)
            model.train()
            
            ppl = math.exp(min(avg_loss, 20))
            print(f"  Step {step:4d}: Train {avg_loss:.4f}, Val {val_loss:.4f}, PPL {ppl:.2f}")
            
            running_loss = 0.0
            n_loss = 0
    
    metrics['time'] = time.time() - start_time
    
    # Final stats
    print(f"\n  Final: Train {metrics['train_losses'][-1]:.4f}, Val {metrics['val_losses'][-1]:.4f}")
    print(f"  Time: {metrics['time']:.1f}s, Throughput: {metrics['tokens']/metrics['time']:.0f} tok/s")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return metrics


def main():
    print("="*60)
    print("FUSION VARIANT BENCHMARK - Survival of the Fittest")
    print("="*60)
    print(f"\nConfig:")
    print(f"  Steps: {CONFIG['max_steps']}")
    print(f"  Batch: {CONFIG['batch_size']}")
    print(f"  Seq: {CONFIG['seq_len']}")
    print(f"  LR: {CONFIG['lr']}")
    print(f"  AMP: {CONFIG['use_amp']}")
    print(f"  Device: {device}")
    
    # Load data once
    print("\nLoading data...")
    dataset, tokenizer = load_stateful_dataset(
        filepath="shakespeare.txt",
        batch_size=CONFIG['batch_size'],
        seq_len=CONFIG['seq_len'],
    )
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Run all variants
    results = {}
    for name, desc in VARIANTS.items():
        results[name] = train_variant(name, desc, dataset, vocab_size)
    
    # ============ LEADERBOARD ============
    print("\n" + "="*70)
    print("LEADERBOARD - Survival of the Fittest")
    print("="*70)
    
    # Sort by final val loss (lower = better)
    ranked = sorted(results.items(), key=lambda x: x[1]['val_losses'][-1])
    
    print(f"\n{'Rank':<5} {'Variant':<6} {'Final Val':<10} {'Final Train':<12} {'PPL':<8} {'Params':<12} {'Time':<8}")
    print("-"*70)
    
    for i, (name, m) in enumerate(ranked, 1):
        final_train = m['train_losses'][-1]
        final_val = m['val_losses'][-1]
        ppl = math.exp(min(final_val, 20))
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal}{i:<4} {name:<6} {final_val:<10.4f} {final_train:<12.4f} {ppl:<8.2f} {m['n_params']:<12,} {m['time']:<.1f}s")
    
    # Winner
    winner = ranked[0][0]
    winner_desc = VARIANTS[winner][0]
    print(f"\n{'='*70}")
    print(f"🏆 WINNER: {winner} ({winner_desc})")
    print(f"   Final Val Loss: {results[winner]['val_losses'][-1]:.4f}")
    print(f"   Improvement over last: {results[ranked[-1][0]]['val_losses'][-1] - results[winner]['val_losses'][-1]:.4f}")
    print(f"{'='*70}")
    
    # Loss reduction comparison
    print("\n📊 LOSS REDUCTION (Start → End):")
    for name, (desc, _) in VARIANTS.items():
        m = results[name]
        reduction = ((m['val_losses'][0] - m['val_losses'][-1]) / m['val_losses'][0]) * 100
        print(f"  {name}: {m['val_losses'][0]:.2f} → {m['val_losses'][-1]:.2f} ({reduction:+.1f}%)")
    
    # Convergence speed (steps to reach certain loss)
    print("\n⚡ CONVERGENCE SPEED (Steps to reach Val Loss < 3.0):")
    for name, m in ranked:
        reached = None
        for i, val in enumerate(m['val_losses']):
            if val < 3.0:
                reached = m['steps'][i]
                break
        if reached:
            print(f"  {name}: {reached} steps")
        else:
            print(f"  {name}: Did not reach (final: {m['val_losses'][-1]:.2f})")


if __name__ == "__main__":
    main()
