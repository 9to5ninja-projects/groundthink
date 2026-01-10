"""
Benchmark Suite for GroundThink V4

Reusable performance benchmarks with fixed configs for reproducible comparisons.
Run: python benchmark_suite.py [--batch 8] [--seq 64] [--steps 100]

Benchmarks:
  B1: Throughput (tok/s at fixed batch/seq)
  B2: Memory (peak VRAM at fixed config)
  B3: Stability (loss delta over training steps)
"""

import os
# Set compiler environment variables FIRST
os.environ['CXX'] = '/usr/bin/g++'
os.environ['CC'] = '/usr/bin/gcc'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from hybrid_v4 import create_hybrid_5m
from data_loader import load_stateful_dataset


# ============================================================================
# BENCHMARK CONFIGURATION (Fixed for reproducibility)
# ============================================================================

DEFAULT_CONFIG = {
    'batch_size': 8,
    'seq_len': 64,
    'steps': 100,
    'warmup_steps': 5,
    'lr': 3e-4,
}


# ============================================================================
# B1: THROUGHPUT BENCHMARK
# ============================================================================

def benchmark_throughput(model, dataset, batch_size, seq_len, warmup=5, trials=20):
    """
    Measure throughput in tokens per second.
    
    Returns:
        dict with tok_per_sec, avg_ms, std_ms
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get sample batches
    data_iter = iter(dataset)
    batches = []
    for _ in range(warmup + trials):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            batch = next(data_iter)
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        batches.append(x)
    
    # Warmup
    with torch.no_grad():
        for i in range(warmup):
            _ = model(batches[i])
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    torch.cuda.synchronize()
    
    for i in range(trials):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(batches[warmup + i])
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    import numpy as np
    times = np.array(times)
    avg_ms = times.mean()
    std_ms = times.std()
    tokens_per_batch = batch_size * seq_len
    tok_per_sec = tokens_per_batch / (avg_ms / 1000)
    
    return {
        'tok_per_sec': tok_per_sec,
        'avg_ms': avg_ms,
        'std_ms': std_ms,
        'batch_size': batch_size,
        'seq_len': seq_len,
    }


# ============================================================================
# B2: MEMORY BENCHMARK
# ============================================================================

def benchmark_memory(model, batch_size, seq_len, vocab_size=None):
    """
    Measure peak VRAM usage.
    
    Returns:
        dict with peak_vram_mb, allocated_mb
    """
    device = next(model.parameters()).device
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Infer vocab_size from model if not provided
    if vocab_size is None:
        vocab_size = model.embed.num_embeddings
    
    # Create input tensor (token IDs)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass
    with torch.no_grad():
        _ = model(x)
    
    torch.cuda.synchronize()
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MiB
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MiB
    
    return {
        'peak_vram_mb': peak_vram,
        'allocated_mb': allocated,
        'batch_size': batch_size,
        'seq_len': seq_len,
    }


# ============================================================================
# B3: STABILITY BENCHMARK (Mini Training Run)
# ============================================================================

def benchmark_stability(model, dataset, steps=100, lr=3e-4, batch_size=8, seq_len=64, use_amp=False):
    """
    Run mini training to verify loss decreases.
    
    Returns:
        dict with loss_start, loss_end, loss_delta, avg_grad_norm, passed
    """
    model.train()
    device = next(model.parameters()).device
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler('cuda') if use_amp else None
    data_iter = iter(dataset)
    
    losses = []
    grad_norms = []
    
    for step in range(steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            batch = next(data_iter)
        
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        targets = x[:, 1:].contiguous()
        inputs = x[:, :-1].contiguous()
        
        # Forward (with optional AMP)
        optimizer.zero_grad()
        with autocast('cuda', enabled=use_amp):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        # Backward (with optional scaler)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        losses.append(loss.item())
        grad_norms.append(grad_norm.item())
    
    loss_start = losses[0]
    loss_end = losses[-1]
    loss_delta = loss_end - loss_start
    avg_grad_norm = sum(grad_norms) / len(grad_norms)
    
    # Pass if loss decreased
    passed = loss_delta < 0
    
    return {
        'loss_start': loss_start,
        'loss_end': loss_end,
        'loss_delta': loss_delta,
        'avg_grad_norm': avg_grad_norm,
        'steps': steps,
        'passed': passed,
    }


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def print_report(b1, b2, b3, config):
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUITE REPORT")
    print("=" * 70)
    
    print(f"\nConfig: batch={config['batch_size']}, seq={config['seq_len']}, steps={config['steps']}")
    
    print("\n--- B1: THROUGHPUT ---")
    print(f"  Tokens/sec: {b1['tok_per_sec']:,.0f}")
    print(f"  Avg time:   {b1['avg_ms']:.2f} ms ± {b1['std_ms']:.2f} ms")
    
    print("\n--- B2: MEMORY ---")
    print(f"  Peak VRAM:  {b2['peak_vram_mb']:.1f} MiB")
    print(f"  Allocated:  {b2['allocated_mb']:.1f} MiB")
    
    print("\n--- B3: STABILITY ---")
    print(f"  Loss start: {b3['loss_start']:.4f}")
    print(f"  Loss end:   {b3['loss_end']:.4f}")
    print(f"  Loss delta: {b3['loss_delta']:+.4f}")
    print(f"  Avg grad:   {b3['avg_grad_norm']:.4f}")
    status = "✅ PASS" if b3['passed'] else "❌ FAIL"
    print(f"  Status:     {status} (loss {'decreased' if b3['passed'] else 'increased'})")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  B1 Throughput: {b1['tok_per_sec']:,.0f} tok/s")
    print(f"  B2 Memory:     {b2['peak_vram_mb']:.1f} MiB")
    print(f"  B3 Stability:  {status}")
    print("=" * 70)
    
    return {
        'config': config,
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'all_passed': b3['passed'],
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GroundThink V4 Benchmark Suite')
    parser.add_argument('--batch', type=int, default=DEFAULT_CONFIG['batch_size'], 
                        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})")
    parser.add_argument('--seq', type=int, default=DEFAULT_CONFIG['seq_len'],
                        help=f"Sequence length (default: {DEFAULT_CONFIG['seq_len']})")
    parser.add_argument('--steps', type=int, default=DEFAULT_CONFIG['steps'],
                        help=f"Training steps for B3 (default: {DEFAULT_CONFIG['steps']})")
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                        help=f"Learning rate for B3 (default: {DEFAULT_CONFIG['lr']})")
    parser.add_argument('--amp', action='store_true',
                        help="Enable automatic mixed precision (AMP)")
    parser.add_argument('--compile', action='store_true',
                        help="Enable torch.compile optimization")
    args = parser.parse_args()
    
    config = {
        'batch_size': args.batch,
        'seq_len': args.seq,
        'steps': args.steps,
        'lr': args.lr,
        'use_amp': args.amp,
        'use_compile': args.compile,
    }
    
    print("=" * 70)
    print("GROUNDTHINK V4 BENCHMARK SUITE")
    print("=" * 70)
    print(f"\nConfig: batch={config['batch_size']}, seq={config['seq_len']}, steps={config['steps']}")
    if config['use_amp']:
        print("Mode: Mixed Precision (AMP)")
    if config['use_compile']:
        print("Mode: torch.compile enabled")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = create_hybrid_5m().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    
    # Apply torch.compile if requested
    if config['use_compile']:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Load dataset
    print("Loading dataset...")
    dataset, tokenizer = load_stateful_dataset(
        'shakespeare.txt',
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        scale='8M',
    )
    
    # Run benchmarks
    print("\n--- Running B1: Throughput ---")
    b1 = benchmark_throughput(model, dataset, config['batch_size'], config['seq_len'])
    print(f"  → {b1['tok_per_sec']:,.0f} tok/s")
    
    print("\n--- Running B2: Memory ---")
    b2 = benchmark_memory(model, config['batch_size'], config['seq_len'])
    print(f"  → {b2['peak_vram_mb']:.1f} MiB peak VRAM")
    
    print("\n--- Running B3: Stability ({} steps) ---".format(config['steps']))
    b3 = benchmark_stability(model, dataset, config['steps'], config['lr'], 
                             config['batch_size'], config['seq_len'], config['use_amp'])
    print(f"  → Loss: {b3['loss_start']:.4f} → {b3['loss_end']:.4f} ({b3['loss_delta']:+.4f})")
    
    # Print report
    results = print_report(b1, b2, b3, config)
