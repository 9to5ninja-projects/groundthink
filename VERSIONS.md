# GroundThink Model Versions

**⚠️ LEGACY REFERENCE** — This file documents V0.1-0.2 (superseded). Current version is **V4.2-Alpha**. See [CHANGELOG.md](CHANGELOG.md) for V4 version history.

## Version Naming Convention
`groundthink_v{major}.{minor}.{patch}_{params}_{tokens}`

- **major**: Architecture changes
- **minor**: Hyperparameter/balance changes  
- **patch**: Training run iterations
- **params**: Parameter count (5M, 125M, etc.)
- **tokens**: Training tokens seen

---

## v0.1.0_5M - Baseline (2026-01-08)

### Architecture
- 6 layers, 256 dim, 8 heads, 32 head_dim
- 5.5M parameters
- FLA `chunk_simple_gla` backend

### Training
- Dataset: 199k samples (~31M tokens), clean deduplicated
- Mix: 57% TinyStories, 43% Gutenberg classics
- Steps: 5000, batch 16, seq 256
- LR: base 3e-4, decay 1e-3 (per-component)
- Warmup: 500 steps, cosine decay

### Results
- Final loss: 0.80
- Speed: 52.7k tok/s
- State norms: Stable at 90.5 (normalization working)

### Balance Analysis
```
RWKV Grounding (base_decay):
  Average retention: 57.8%
  → BALANCED

Mamba Selectivity (time_decay projection):
  Average norm: 195.9
  → HIGH SELECTIVITY (potentially too high)
```

### Issues Identified
1. **Mamba-dominant**: time_decay projection norm (196) overwhelms grounding
2. **Short prompts fail**: Model needs context to generate
3. **Mixed style output**: TinyStories patterns bleeding into literary prompts

### Files
- Model: `groundthink_v0.1.0_5M.pt`
- Training: `train_hybrid_v2.py`
- Layers: `layers.py`

---

## v0.2.0_5M - Rebalanced (2026-01-08)

### Changes from v0.1.0
- **Balance alpha**: 0.6 (more RWKV grounding)
- **Combination formula**: Linear interpolation
  - `w_combined = alpha * w_base + (1 - alpha) * w_selective`

### Training
- Same dataset: 199k samples (~31M tokens)
- Same LR schedule: base 3e-4, decay 1e-3
- 5000 steps, batch 16, seq 256

### Results
- Final loss: 0.87 (slightly worse than v0.1.0's 0.80)
- Speed: 52k tok/s
- State norms: Stable at 90.5 throughout

### Generation Quality
- "Once upon a time" → Coherent TinyStories output
- Other prompts → Still poor (expected at this scale)

### Analysis
- More grounding (alpha=0.6) slowed learning slightly
- States perfectly stable - normalization working
- Need more training tokens for non-story prompts

### Files
- Model: `groundthink_v020_5M.pt`
- Training: `train_v020.py`
- Layers: `layers_v020.py`

---

## v0.2.0_5M_10k - Extended Training (2026-01-08)

### Purpose
Test effect of 2x longer training on same data.

### Results
- Final loss: 0.79 (down from 0.87 at 5k)
- Best loss: 0.77
- Speed: 52k tok/s
- States: 90.5 stable throughout

### Generation Quality
- "Once upon a time" → Cleaner TinyStories output
- Non-story prompts → Still fragments (need more diverse data)

### Analysis
- Loss improved 10% with 2x training
- Model still learning at 10k (curve not flat)
- Non-story generation weak → need data diversity, not more epochs

---

## Experiment Log

| Version | Date | Loss | Notes |
|---------|------|------|-------|
| v0.1.0 | 2026-01-08 | 0.80 | Baseline, Mamba-dominant |
| v0.2.0 | 2026-01-08 | 0.87 | alpha=0.6, slightly slower learning |
| v0.2.0_10k | 2026-01-08 | 0.77 | Extended training, 10% improvement |

---

## Key Learnings

### From Research
- SSMs need 2-4x more data than transformers
- Hybrid architectures need per-component learning rates
- State normalization prevents explosion in deep networks
- HiPPO-style initialization critical for SSMs

### From Our Experiments
- Duplicate data causes 50%+ waste - always deduplicate
- time_decay projection grows large during training - needs regularization
- Base retention 50-60% is healthy for this model size
- State norms should be monitored and normalized
