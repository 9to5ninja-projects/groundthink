# GroundThink V4: Parallel Hybrid RWKV-6 + Mamba-2 Architecture

**Status:** Phase 2 Complete — Winning variant identified and benchmarked  
**Updated:** 2026-01-09  
**Repository:** https://github.com/9to5ninja-projects/groundthink

---

## What is GroundThink V4?

GroundThink V4 is a **parallel hybrid architecture** combining:
- **RWKV-6** (smooth, long-range recurrent attention)
- **Mamba-2** (selective, efficient state-space model)

Both components run **in parallel within each block**, fused via gating mechanism. This design leverages RWKV's memory depth and Mamba's selectivity in a single forward pass.

**Key innovation:** Gated Fusion (GF) learns per-position weighting between the two pathways, enabling the model to context-switch between recurrent and selective modes.

---

## Architecture Overview

### The Building Block: ParallelHybridBlock

```
┌─────────────────────────────────────┐
│       Input: [batch, seq, 128]      │
├─────────────────────────────────────┤
│                                     │
│  Norm                               │
│  ├─→ RWKV-6 ──┐                    │
│  └─→ Mamba-2 ─┤                    │
│               ▼                     │
│         Gated Fusion (learns α)    │
│         output = α·rwkv + (1-α)·mamba
│               │                     │
│               + SKIP ────────────→  │
│               │                     │
│               ▼                     │
│         RMSNorm + FFN              │
│               │                     │
│               + SKIP ────────────→  │
│               │                     │
│               ▼                     │
│     Output: [batch, seq, 128]      │
│                                     │
└─────────────────────────────────────┘
```

**See [V4_DESIGN.md](V4_DESIGN.md) for detailed architecture diagrams and layer specifications.**

### Full Model Architecture

**GF-MH (Phase 2 Winner):**
- 8 ParallelHybridBlocks stacked
- Each block: 1 RWKV-6 (∥) 1 Mamba-2 in parallel
- Gated Fusion with gate_init=0.3 (favors Mamba)
- ~3.5M total parameters
- vocab_size=97 (Shakespeare character tokenizer)

---

## Phase 2 Results: Fusion & Ratio Benchmarking

### Fusion Strategy Comparison (5 variants)

| Rank | Strategy | Model | Val Loss | Improvement | Throughput |
|------|----------|-------|----------|------------|-----------|
| 🥇 **WINNER** | **Gated Fusion** | **GF** | **1.6891** | **-4% vs HY** | 42.9K tok/s |
| 2 | Concat+Project | CP | 1.6919 | -3.8% | 47.7K tok/s |
| 3 | Baseline | HY | 1.7600 | — | 31.7K tok/s |
| 4 | Weighted Sum | WS | 1.8185 | +3.3% | 45.4K tok/s |
| 5 | Residual Fusion | RF | 1.9480 | +10.6% | 47.4K tok/s |

**Finding:** Gated fusion with learnable per-position weighting outperforms all alternatives.

### Ratio Strategy Comparison (3 variants of GF)

| Rank | Component Balance | Model | Val Loss | vs GF Baseline |
|------|-------------------|-------|----------|----------------|
| 🥇 **OVERALL WINNER** | **Mamba-Heavy (70%)** | **GF-MH** | **1.6700** | **-1.8%** |
| 2 | Balanced (50-50) | GF | 1.6998 | — |
| 3 | RWKV-Heavy (70%) | GF-RH | 1.7201 | +0.3% |

**Finding:** Mamba-selective capabilities benefit from higher relative weight. RWKV-Heavy performs worse.

### Implementation Details

**All variants tested with:**
- Training: 500 steps, batch_size=64, seq_len=64
- Optimizer: AdamW, lr=3e-4, warmup=100 steps
- Dataset: shakespeare.txt (97 vocab, char-level tokenization)
- Validation: Loss computed every 50 steps

**All model code in:** [hybrid_v4_ratio.py](hybrid_v4_ratio.py) (GF-MH final implementation)

---

## Quick Start: Running Benchmarks

### Requirements

```bash
# Install dependencies (Python 3.10+, CUDA 12.1+)
pip install -r requirements.txt

# On Linux, install optional faster kernels
pip install causal-conv1d mamba-ssm
```

### Run All Variant Benchmarks (5 Minute Quickstart)

**See [GETTING_STARTED.md](GETTING_STARTED.md) for step-by-step instructions.**

```bash
# 1. Setup
git clone https://github.com/9to5ninja-projects/groundthink.git
cd groundthink
source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify
python -m tests.test_phase0_complete

# 3. Run benchmark
python benchmark_variants.py
```

### Test Individual Variant

```python
import torch
from models import get_model, list_models

# List available models
list_models(show=True)  # Shows: tiny (0.5M), small (3.6M), medium (7.9M), etc.

# Load a model
model = get_model('medium')  # or 'small', 'tiny', 'GF-MH', etc.
model = model.to('cuda')

# Forward pass
x = torch.randint(0, 97, (4, 64), device='cuda')  # batch=4, seq=64
logits = model(x)
print(f"Output shape: {logits.shape}")  # Should be [4, 64, 97]
```

---

## Documentation Map

**Start here (in order by your goal):**
1. **[ONBOARDING.md](ONBOARDING.md)** — What are RWKV and Mamba? Why combine them? (Conceptual foundation)
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** — Clone, install, run first benchmark (5 minutes)
3. **[README.md](README.md)** — This file: quick reference and Phase 2 results
4. **[V4_DESIGN.md](V4_DESIGN.md)** — Architecture specification and implementation details

**For specific needs:**
- **[DATA_FLOW.md](DATA_FLOW.md)** — Visual architecture diagrams (6 levels from token to output)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — How to add variants, report bugs, contribute improvements
- **[V4_STRATEGY.md](V4_STRATEGY.md)** — Task backlog, complexity assessment, validation gates
- **[V4_HANDOFF.md](V4_HANDOFF.md)** — Current status, audit summary, git approval protocol
- **[CHANGELOG.md](CHANGELOG.md)** — Version history with dates and major changes
- **[VERSION](VERSION)** — Current semantic version
- **[V4_TRAINING_GUIDE.md](V4_TRAINING_GUIDE.md)** — Training procedures and hyperparameter tuning
- **[V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md)** — Performance optimization & monitoring

**Legacy/Reference:**
- **[README_A100.md](README_A100.md)** — A100 cloud training setup (legacy)
- **[VERSIONS.md](VERSIONS.md)** — Old v0.1-0.2 version records (reference only)

---

## Next Steps: Phase 3 (Scaling)

After Phase 2 completion, the next phase involves:

| Task | Goal | Status |
|------|------|--------|
| **Task 19** | Scale GF-MH to 8M parameters | ⬜ PENDING |
| **Task 20** | Extended training (50K steps) | ⬜ PENDING |
| **Task 21** | Needle-in-a-Haystack (NIAH) test | ⬜ PENDING |

See [V4_STRATEGY.md](V4_STRATEGY.md#phase-3-scale-testing-after-phase-2) for full Phase 3 details.

---

## Key Parameters (3.5M Model)

```python
model_config = {
    'vocab_size': 97,              # Shakespeare char tokenizer
    'hidden_size': 128,            # Embedding/layer dim
    'n_layers': 8,                 # Parallel hybrid blocks
    'n_heads': 8,                  # RWKV6 attention heads
    
    # Layer counts
    'n_rwkv': 8,                   # One per block (parallel)
    'n_mamba': 8,                  # One per block (parallel)
    
    # Mamba2 specific
    'mamba_expand': 2,             # Internal expansion ratio
    'mamba_head_dim': 64,          # Per-head dimension
    
    # GF-MH specific (Phase 2 winner)
    'fusion': 'gated',             # Learned per-position weighting
    'gate_init': 0.3,              # Initial Mamba bias (0.3 = 70% Mamba)
}
```

**Parameter breakdown:**
- RWKV-6 (8 layers × 128 hidden): ~5.6M params
- Mamba-2 (8 layers × 128 hidden): ~536K params
- Embedding + Output head: ~1.28M (tied)
- **Total: ~3.5M params**

---

## For Developers

### File Structure

```
groundthink/
├── models/                      # Model implementations
│   ├── __init__.py              # Registry: get_model('medium'), list_models()
│   ├── hybrid_v4_ratio.py       # Phase 2 winner (GF-MH)
│   ├── hybrid_v4_GF.py          # Gated Fusion variant
│   ├── hybrid_v4_CP.py          # Concat+Project variant
│   ├── hybrid_v4_WS.py          # Weighted Sum variant
│   ├── hybrid_v4_RF.py          # Residual Fusion variant
│   └── hybrid_v4.py             # Baseline (HY) variant
│
├── data/                        # Data loading & tokenization
│   ├── __init__.py              # load_stateful_dataset()
│   ├── data_loader.py           # Shakespeare dataset
│   ├── tokenizer.py             # Character-level tokenizer
│   └── shakespeare.txt          # Training data
│
├── configs/                     # Training configurations (YAML)
│   ├── train_medium_50k.yaml    # 50K step extended training
│   ├── train_quick.yaml         # Fast validation (500 steps)
│   └── train_default.yaml       # Standard training
│
├── checkpoints/                 # Saved model weights (gitignored)
│
├── tests/                       # Test suite
│   ├── test_phase0_complete.py  # Environment validation
│   └── test_monitoring.py       # Monitoring tests
│
├── benchmark_variants.py        # Comprehensive benchmark suite
├── train_v4.py                  # V4 training script
└── train.py                     # Training loop (legacy)
```

### Adding a New Variant

To test a new fusion strategy:

1. Create `hybrid_v4_XXXX.py` copying from [hybrid_v4_GF.py](hybrid_v4_GF.py)
2. Modify the `fuse()` method with your strategy
3. Register in [benchmark_variants.py](benchmark_variants.py) variants dict
4. Run `python benchmark_variants.py`

---

## Training Details

### Config for 6GB VRAM

```python
# See V4_DESIGN.md Section "Training Configuration"
training_config = {
    'batch_size': 32,
    'grad_accum_steps': 2,         # Effective batch ~64
    'max_seq_len': 256,
    'lr': 3e-4,
    'warmup_steps': 200,
}
```

### Memory Breakdown (3.5M Model @ batch=64, seq=64)

| Component | Size |
|-----------|------|
| Model weights (FP32) | 14 MB |
| Gradients | 14 MB |
| Optimizer states (Adam) | 56 MB |
| Batch activations | ~100 MB |
| **Total** | **~184 MB** |

Easily fits in 6GB VRAM with room for longer sequences or larger batches.

---

## Contributing

Contributions follow the **survival of the fittest** approach:

1. Create a new variant (fork hybrid_v4_GF.py)
2. Benchmark it against current winner (GF-MH)
3. If it beats the winner, merge it
4. Update README with new results

The only gate: **must benchmark fairly** (same dataset, same steps, same seeds).

---

## License

[Specify your license here - Apache 2.0? MIT? Commercial?]

---

## Questions?

See documentation in this order:
1. **Architecture:** [V4_DESIGN.md](V4_DESIGN.md)
2. **Tasks & Progress:** [V4_STRATEGY.md](V4_STRATEGY.md)
3. **Current Status:** [V4_HANDOFF.md](V4_HANDOFF.md)
4. **Implementation Details:** Code comments in [hybrid_v4_ratio.py](hybrid_v4_ratio.py)

---

**Last Updated:** 2026-01-09 (Phase 2 Complete)
