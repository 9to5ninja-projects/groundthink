"""
Task 0.0.1: Pure RWKV-6 Baseline Benchmark

GroundThink — Phase 0: Base Model Characterization
Copyright (c) 2026 Matthew [m_tes]

ATTRIBUTION:
Built on top of:
    - RWKV-6 architecture (Peng et al., 2024)
    - WikiText-103 dataset (Merity et al., 2016)
    - Standard training practices from PyTorch ecosystem

OUR CONTRIBUTION:
    - Integrated training + metrics collection framework
    - Variance analysis integration (novel diagnostic)
    - NIAH testing adaptation for BPE tokenization
    - Systematic base model characterization methodology (Phase 0)

See ATTRIBUTION.md for full citation details.

Integrated training + metrics collection for base model characterization.
Extends train_v4.py monitoring with variance analysis and NIAH testing.

Metrics collected:
- PRIMARY: Perplexity, throughput (tokens/sec), memory (GB)
- STATE DYNAMICS: State magnitude, norm stability
- GRADIENT ANALYSIS: Gradient norms, variance across layers
- INFORMATION FLOW: Long-context retrieval (NIAH), ablation sensitivity

Results saved to: logs/task_0_0_1/BASE_MODEL_FINDINGS_RWKV6.md
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import yaml
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# Import pure RWKV-6 model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rwkv6_pure import create_rwkv6_4m
from data import load_stateful_dataset


def load_config(config_path: str) -> dict:
    """Load YAML config for Task 0.0.1"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Cosine LR schedule with warmup"""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    elif step > max_steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)


def compute_metrics(model, batch, device):
    """
    Compute primary metrics (perplexity, throughput, memory).
    
    Returns:
        dict: {
            'loss': float,
            'ppl': float,
            'throughput': float (tokens/sec),
            'memory_gb': float,
        }
    """
    x, y = batch
    x, y = x.to(device), y.to(device)
    
    # Measure throughput
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time
    
    # Metrics
    batch_size, seq_len = x.shape
    num_tokens = batch_size * seq_len
    throughput = num_tokens / elapsed
    
    memory_gb = 0.0
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
    
    return {
        'loss': loss.item(),
        'ppl': math.exp(loss.item()),
        'throughput': throughput,
        'memory_gb': memory_gb,
    }


def compute_gradient_metrics(model):
    """
    Compute gradient health metrics.
    
    Returns:
        dict: {
            'grad_norm': float,
            'grad_std': float,
            'layer_grad_norms': list[float],
        }
    """
    total_norm = 0.0
    layer_norms = []
    all_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            
            # Track per-layer norms
            if 'blocks' in name:
                layer_idx = int(name.split('.')[1])
                if layer_idx >= len(layer_norms):
                    layer_norms.extend([0.0] * (layer_idx - len(layer_norms) + 1))
                layer_norms[layer_idx] += param_norm ** 2
            
            # Collect all gradient values for variance
            all_grads.extend(param.grad.data.flatten().cpu().numpy())
    
    total_norm = math.sqrt(total_norm)
    layer_norms = [math.sqrt(n) for n in layer_norms]
    
    import numpy as np
    grad_std = float(np.std(all_grads)) if all_grads else 0.0
    
    return {
        'grad_norm': total_norm,
        'grad_std': grad_std,
        'layer_grad_norms': layer_norms,
    }


def compute_state_metrics(model, batch, device):
    """
    Compute state dynamics metrics (magnitude, stability).
    
    Note: RWKV-6 uses FLA's internal state management.
    This is a placeholder for future state extraction.
    
    Returns:
        dict: {
            'state_norm': float,
            'state_std': float,
        }
    """
    # TODO: Extract RWKV-6 states from FLA wrapper
    # For now, return placeholders
    return {
        'state_norm': 0.0,
        'state_std': 0.0,
    }


def save_checkpoint(model, optimizer, step, config, checkpoint_dir):
    """Save training checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"ckpt_{step:06d}.pt"
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, checkpoint_path)
    
    print(f"Saved checkpoint: {checkpoint_path}")


def generate_report(metrics_history, config, output_path):
    """
    Generate BASE_MODEL_FINDINGS_RWKV6.md report.
    
    Args:
        metrics_history: List of metric dicts per step
        config: Training configuration
        output_path: Path to save markdown report
    """
    import numpy as np
    
    # Aggregate metrics
    final_metrics = metrics_history[-1] if metrics_history else {}
    
    losses = [m['loss'] for m in metrics_history if 'loss' in m]
    ppls = [m['ppl'] for m in metrics_history if 'ppl' in m]
    throughputs = [m['throughput'] for m in metrics_history if 'throughput' in m]
    
    # Compute averages
    avg_loss = np.mean(losses[-100:]) if losses else 0.0
    avg_ppl = np.mean(ppls[-100:]) if ppls else 0.0
    avg_throughput = np.mean(throughputs) if throughputs else 0.0
    
    # Generate markdown
    report = f"""# Task 0.0.1: Pure RWKV-6 Baseline Findings

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: Pure RWKV-6 (8 layers × 144 hidden, 4.46M params)
**Dataset**: WikiText-103 (BPE 16K tokenization)
**Training Steps**: {len(metrics_history)}

## Configuration

```yaml
{yaml.dump(config, default_flow_style=False)}
```

## Primary Metrics

| Metric | Final Value | Mean (last 100 steps) |
|--------|-------------|------------------------|
| Loss | {final_metrics.get('loss', 0.0):.4f} | {avg_loss:.4f} |
| Perplexity | {final_metrics.get('ppl', 0.0):.2f} | {avg_ppl:.2f} |
| Throughput | {final_metrics.get('throughput', 0.0):.1f} tok/s | {avg_throughput:.1f} tok/s |
| Memory | {final_metrics.get('memory_gb', 0.0):.2f} GB | - |

## Gradient Health

| Metric | Final Value |
|--------|-------------|
| Grad Norm | {final_metrics.get('grad_norm', 0.0):.4f} |
| Grad Std | {final_metrics.get('grad_std', 0.0):.6f} |

### Layer-wise Gradient Norms

```python
{final_metrics.get('layer_grad_norms', [])}
```

## State Dynamics

*(State extraction not yet implemented for FLA RWKV-6)*

| Metric | Final Value |
|--------|-------------|
| State Norm | {final_metrics.get('state_norm', 0.0):.4f} |
| State Std | {final_metrics.get('state_std', 0.0):.6f} |

## Observations

### Convergence Behavior

- Final loss: {final_metrics.get('loss', 0.0):.4f} (ppl: {final_metrics.get('ppl', 0.0):.2f})
- Training stable: {'✓' if avg_loss < 10.0 else '✗'}
- Gradient health: {'✓' if final_metrics.get('grad_norm', 0.0) < 100.0 else '✗'}

### Performance Characteristics

- Throughput: {avg_throughput:.1f} tokens/sec
- Memory footprint: {final_metrics.get('memory_gb', 0.0):.2f} GB

### Questions for Comparison

1. **Is RWKV-6 a stabilizer or destabilizer?**
   - Gradient variance: {final_metrics.get('grad_std', 0.0):.6f}
   - (Low variance → stabilizer, High variance → destabilizer)

2. **Information flow quality?**
   - NIAH test results: *(See test_niah_bpe.py)*
   - Long-context retention: *(Pending)*

3. **Gradient efficiency?**
   - Gradient norm: {final_metrics.get('grad_norm', 0.0):.4f}
   - Compare with Mamba-2 in Task 0.0.2

## Next Steps

- [ ] Run Task 0.0.2 (Pure Mamba-2 baseline)
- [ ] Compare RWKV-6 vs Mamba-2 characteristics
- [ ] Execute NIAH test at checkpoints
- [ ] Complete variance analysis tool

---

*Generated by tests/task_0_0_1_rwkv6_benchmark.py*
"""
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved: {output_path}")


def main():
    """Main training loop for Task 0.0.1"""
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'task_0_0_1.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please create configs/task_0_0_1.yaml first")
        return
    
    config = load_config(config_path)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading WikiText-103 (BPE 16K)...")
    dataset, tokenizer = load_stateful_dataset(
        'data/wikitext-103',
        batch_size=config['batch_size'],
        seq_len=config['seq_len'],
        scale='LARGE',  # Forces BPE
    )
    
    # Create model
    print(f"\nCreating pure RWKV-6 model (vocab={tokenizer.vocab_size})...")
    model = create_rwkv6_4m(vocab_size=tokenizer.vocab_size).to(device)
    print(f"Model params: {model.get_num_params():,} total")
    print(f"Non-embedding params: {model.get_num_params(non_embedding=True):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=config['betas'],
    )
    
    # Training state
    warmup_steps = int(config['warmup_ratio'] * config['max_steps'])
    metrics_history = []
    checkpoint_dir = Path('checkpoints') / 'task_0_0_1'
    
    print(f"\nStarting training...")
    print(f"Max steps: {config['max_steps']}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Eval every: {config['eval_every']}")
    print(f"Save every: {config['save_every']}")
    print()
    
    # Training loop
    model.train()
    step = 0
    
    while step < config['max_steps']:
        for batch in dataset:
            step += 1
            
            # Adjust learning rate
            lr = get_lr(step, warmup_steps, config['max_steps'], config['lr'], config['min_lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward + backward
            optimizer.zero_grad()
            
            metrics = compute_metrics(model, batch, device)
            loss = torch.tensor(metrics['loss'], requires_grad=True, device=device)
            
            # Recompute for gradient
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            
            # Gradient metrics
            grad_metrics = compute_gradient_metrics(model)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            
            # State metrics (placeholder)
            state_metrics = compute_state_metrics(model, batch, device)
            
            # Merge metrics
            metrics.update(grad_metrics)
            metrics.update(state_metrics)
            metrics['lr'] = lr
            metrics['step'] = step
            
            metrics_history.append(metrics)
            
            # Logging
            if step % config['log_every'] == 0:
                print(f"Step {step:5d} | Loss {metrics['loss']:.4f} | PPL {metrics['ppl']:6.2f} | "
                      f"LR {lr:.2e} | Grad {metrics['grad_norm']:.3f} | "
                      f"Tok/s {metrics['throughput']:.0f}")
            
            # Checkpointing
            if step % config['save_every'] == 0:
                save_checkpoint(model, optimizer, step, config, checkpoint_dir)
            
            # Max steps reached
            if step >= config['max_steps']:
                break
    
    # Final checkpoint
    save_checkpoint(model, optimizer, step, config, checkpoint_dir)
    
    # Generate report
    report_path = Path('logs') / 'task_0_0_1' / 'BASE_MODEL_FINDINGS_RWKV6.md'
    generate_report(metrics_history, config, report_path)
    
    print(f"\n{'='*60}")
    print(f"Task 0.0.1 complete!")
    print(f"Final loss: {metrics_history[-1]['loss']:.4f}")
    print(f"Final PPL: {metrics_history[-1]['ppl']:.2f}")
    print(f"Report: {report_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
