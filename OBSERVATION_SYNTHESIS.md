# Observation Synthesis — 2026-01-11

**Purpose:** Consolidate all 18 observations into actionable inferences before revisiting base models.

---

## Executive Summary

After 18 observations across Tasks 42-62, we have a clear picture:

| Finding | Evidence | Confidence |
|---------|----------|------------|
| **RWKV dominates by design** | Obs 13, 16, Task 58 | HIGH |
| **Mamba contributes marginally but measurably** | D2/D4 PASS, Obs 16 | HIGH |
| **Attractor zone exists at 10-30% R/M ratio** | Obs 14 | HIGH |
| **BPE tokenization improves behavior** | Obs 12, Task 62 | HIGH |
| **GF-MH matches GPT-2 at 17% fewer params** | Task 62 | HIGH |
| **Batch size ≥16 closes speed gap to 2.1x** | Obs 18 | HIGH |
| **Long-context is stable (1.04x at 512 tokens)** | Task 60 | MEDIUM |

---

## Observation Index

| # | Title | Key Data | Inference |
|---|-------|----------|-----------|
| 1 | Per-Dimension Gating | HGF preserves Mamba | More granular gates help balance |
| 2 | Loss vs Balance Inverse | Lower loss → more imbalance | Optimizer trades balance for performance |
| 3 | More Fusion Params ≠ Balance | CP has most params, worst balance | Architecture matters more than param count |
| 4 | Gate Init Doesn't Matter | All converge to 0.06-0.27 | Attractor behavior confirmed |
| 5 | HGF Best for Balance | Ratio 2.15 (others 0.15-0.18) | Per-dim + per-pos gates work |
| 6 | Activation Variance | RWKV var >> Mamba var | State norms explain imbalance |
| 7 | R/M Ratio Drifts Over Time | 0.3→0.7 during training | Optimizer pushes toward RWKV |
| 8 | Higher Mamba LR Worse | Tried 2x, 5x LR | Gradient magnitude isn't the issue |
| 9 | V3-Compliant Warmup | Implemented | No effect on imbalance |
| 10 | Mamba Grad Scaling 10x | FAILED | More gradients ≠ more contribution |
| 11 | RWKV Dropout 40%→10% | FAILED | Forcing Mamba hurts loss |
| 12 | BPE Tokenization | val loss 1.26, balanced flow | THE FIX for many issues |
| 13 | State Norms vs Activations | RWKV 2571, Mamba 3.7 | 689x internal state ratio |
| 14 | Extreme Ratio Experiments | All converge to same zone | Loss landscape has attractor |
| 15 | D1-D4 Diagnostics | D1⚠ D2✓ D3⚠ D4✓ | Components functional, imbalanced |
| 16 | Synthesis | See causal chain | RWKV=accumulator, Mamba=selector |
| 17 | GPT-2 Comparison | 1.008 ratio = EQUIVALENT | Architecture validated |
| 18 | CUDA Batch Scaling | batch=16 → 2.1x gap | Memory-bound at small batch |

---

## Key Inferences

### Inference 1: The Imbalance Is Architectural

```
RWKV-6: state_{t+1} = decay * state_t + new_info
        → Accumulates over time
        → State norm grows with sequence
        → Naturally dominates by having more "mass"

Mamba-2: state = selective_scan(filter(input))
        → Filters selectively
        → State stays bounded
        → Naturally smaller contribution
```

**This is not a bug.** It's how the architectures work by design.

### Inference 2: The Attractor Zone (10-30% R/M)

All gate initializations converge to the same equilibrium:
- GF-XM (3% RWKV init) → 25% ratio after training
- GF-MH (30% RWKV init) → 27% ratio after training  
- GF-XR (97% RWKV init) → 27% ratio after training

**The optimizer finds a loss-minimizing basin** where RWKV does heavy lifting and Mamba provides refinement.

### Inference 3: Mamba Paradox Explained

| Metric | Mamba | RWKV |
|--------|-------|------|
| Gradient magnitude | 10x larger | 1x |
| State contribution | 0.1% | 99.9% |
| Loss impact | Marginal | Dominant |

**Why?** Mamba has more "room to move" (less saturated) but moving its params has less effect on loss (small state). It's like pushing hard on a loose hinge — lots of force, little motion on the door.

### Inference 4: BPE Is The Right Tokenization

| Tokenization | Val Loss | Behavior |
|--------------|----------|----------|
| Char-level | 1.58 | More imbalanced, RWKV dominates |
| BPE (16K) | 1.26 | More balanced, cleaner gradients |

BPE creates cleaner token boundaries → Mamba's selectivity becomes more useful.

### Inference 5: Speed Gap Closes With Batch Size

| Batch | Ratio (GF-MH/GPT-2) | GF-MH tok/s |
|-------|---------------------|-------------|
| 1 | 5.1x slower | 5,700 |
| 16 | 2.1x slower | 90,631 |

**RWKV kernel is memory-bound at small batch.** Use batch≥16 for training.

### Inference 6: The Model Works (Despite Imbalance)

| Test | Result | Meaning |
|------|--------|---------|
| D2 (Frozen State) | PASS | Mamba state varies with input |
| D4 (Info Flow) | PASS | Information flows through both |
| D3 (Balance) | FAIL | Imbalanced but functional |
| Long-context (Task 60) | 1.04x PASS | No degradation at 512 tokens |
| GPT-2 comparison | 1.008 EQUIVALENT | Matches transformer quality |

**The hybrid works.** The imbalance is how it works, not why it fails.

---

## Open Questions

| ID | Question | How To Test |
|----|----------|-------------|
| Q1 | Does Mamba help more on specific token types? | Per-token-type D3 analysis |
| Q2 | Is 10-30% optimal or local minimum? | Train 10x longer, check if zone shifts |
| Q3 | Would 2:1 RWKV:Mamba layers work better? | New architecture variant |
| Q4 | Does the ratio change at 8M, 30M scale? | D3 across scales |
| Q5 | What tokens benefit most from Mamba? | Analyze per-position contributions |
| Q6 | Would residual connections help Mamba? | Add residual from Mamba to output |

---

## Recommended Next Steps

### Option A: Investigate Imbalance Further (Exploratory)

1. Run D3 analysis on **specific token categories** (punctuation, rare words, repetition)
2. Try **residual Mamba connection** (add Mamba output directly to residual stream)
3. Test **2:1 RWKV:Mamba architecture** (more RWKV layers, fewer Mamba)

### Option B: Proceed to V5 Gate (Practical)

1. Task 63: CER (Compute-Efficiency Ratio) at 8M scale
2. Task 64: UCW (Useful Context Window) test
3. Task 65: SPS (State Persistence Score) at 5/10/20/50 turns

### Option C: Revisit Base Models (Recommended)

Go back to pure RWKV-6 and pure Mamba-2 to understand:
- How does pure RWKV behave on same WikiText-103 + BPE?
- How does pure Mamba behave?
- What does each contribute in isolation?
- This establishes ground truth before more hybrid experiments.

---

## Tools Available

| Tool | File | What It Measures |
|------|------|------------------|
| Information Flow | `tools/information_flow_tracer.py` | Component contribution % |
| Thresholds | `tools/thresholds.py` | Unified PASS/WARN/FAIL |
| State Metrics | `tools/state_metrics.py` | Norm, variance over time |
| Gradient Coupling | `tools/gradient_coupling.py` | Grad flow to each component |
| Ablation | `tests/test_ablation.py` | Zero component, measure impact |
| State Evolution | `tests/test_state_evolution.py` | Input variety → state response |
| Long-Context | `tests/test_long_context.py` | 64→512 degradation |
| CUDA Profile | `tests/profile_cuda.py` | Speed comparison |

---

## Data Points For Reassessment

When revisiting base models, measure:

1. **Pure RWKV-6** (5M params)
   - Val loss on WikiText-103 + BPE
   - State norm at seq_len=64, 128, 256, 512
   - Speed (tok/s)

2. **Pure Mamba-2** (5M params)
   - Same metrics

3. **GF-MH** (5.6M params)
   - Same metrics (we have: 6.850 val loss)

4. **GPT-2** (6.8M params)
   - Same metrics (we have: 6.798 val loss)

This gives us a **2x2 comparison matrix**: Stateful (RWKV, Mamba) vs Stateless (GPT-2), Hybrid (GF-MH) vs Pure.

---

*Generated: 2026-01-11*  
*Reference: [archive/V4_FUSION_MODELS.md](archive/V4_FUSION_MODELS.md) Observations 1-18*  
*See also: [V4_STRATEGY.md](V4_STRATEGY.md) for executive summary*
