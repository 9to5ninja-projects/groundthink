# GPT-2 Comparison Experiment Protocol

**Experiment ID:** EXP-001  
**Date:** 2026-01-10  
**Investigator:** GroundThink Team  
**Status:** PROTOCOL DEFINED

---

## 1. Hypothesis

**H₀ (Null):** GF-MH (RWKV+Mamba hybrid) achieves the same language modeling loss as GPT-2 at matched parameter count.

**H₁ (Alternative):** GF-MH achieves different loss than GPT-2, with potential advantages in statefulness or efficiency metrics.

**Research Question:** Does the hybrid stateful architecture (GF-MH) provide competitive performance and/or unique advantages compared to a standard transformer (GPT-2)?

---

## 2. Variables

### 2.1 Independent Variable
| Variable | Levels |
|----------|--------|
| Architecture | GF-MH (hybrid), GPT-2 (transformer) |

### 2.2 Dependent Variables (Measured)
| Variable | Metric | Units | Priority |
|----------|--------|-------|----------|
| Validation Loss | Cross-entropy | nats | Primary |
| Training Time | Wall clock | seconds | Secondary |
| Peak Memory | VRAM usage | MB | Secondary |
| Statefulness | D3 contribution ratio | % | Exploratory |
| Inference Speed | Tokens per second | tok/s | Exploratory |

### 2.3 Control Variables (Held Constant)
| Variable | Value | Verification |
|----------|-------|--------------|
| Parameter Count | ~5.6M | Print and verify both models |
| Vocabulary Size | 16,000 | Same tokenizer file |
| Data Source | shakespeare.txt | Same file, same path |
| Batch Order | seed=42 | Save to batch_order.npy |
| Learning Rate | 3e-4 | Config file |
| LR Schedule | Cosine decay | Config file |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) | Config file |
| Weight Decay | 0.1 | Config file |
| Batch Size | 64 | Config file |
| Sequence Length | 64 | Config file |
| Training Steps | 500 | Config file |
| Gradient Clip | 1.0 | Config file |
| Random Seed | 42 | torch.manual_seed(42) |
| Hardware | Same GPU | Run sequentially on same device |

---

## 3. Materials

### 3.1 Models
- **GF-MH:** `models/hybrid_v4_ratio.py` → `create_hybrid_GF_MH_5m()`
- **GPT-2:** `models/gpt2.py` → `create_gpt2_5m()` (TO BE CREATED)

### 3.2 Data
- **Training:** `data/shakespeare.txt` (1.1M characters)
- **Validation:** Last 10% of training data (same split for both)

### 3.3 Config
```yaml
# configs/exp001_comparison.yaml
seed: 42
model: [GF-MH, GPT2]  # Run both
vocab_size: 16000
batch_size: 64
seq_len: 64
lr: 3e-4
min_lr: 3e-5
warmup_ratio: 0.1
max_steps: 500
eval_every: 50
save_every: 500
use_amp: true
```

---

## 4. Procedure

### Phase 1: Setup (Pre-experiment)
1. [ ] Create GPT-2 model matching 5.6M params
2. [ ] Verify param counts: `print(sum(p.numel() for p in model.parameters()))`
3. [ ] Create shared config file
4. [ ] Generate batch_order.npy with seed=42
5. [ ] Verify both models can forward pass same input

### Phase 2: Smoke Tests (Validation)
1. [ ] **Overfit Test:** Train each on 10 samples for 100 steps
   - Expected: Both reach loss < 0.5
   - If fail: Debug model implementation
   
2. [ ] **Gradient Norm Test:** Log gradient norms for 10 steps
   - Expected: Both in range 0.1-10.0
   - If fail: Check initialization

3. [ ] **100-Step Sanity:** Train 100 steps, compare loss curves
   - Expected: Similar slope (within 2x)
   - If fail: Check training loop

### Phase 3: Main Experiment
1. [ ] Set random seed: `torch.manual_seed(42)`
2. [ ] Load batch order from batch_order.npy
3. [ ] Train GF-MH for 500 steps, log all metrics
4. [ ] Reset random seed: `torch.manual_seed(42)`
5. [ ] Train GPT-2 for 500 steps, log all metrics
6. [ ] Save checkpoints for both

### Phase 4: Measurement
1. [ ] Extract final validation loss from logs
2. [ ] Record training time from timestamps
3. [ ] Record peak memory from torch.cuda.max_memory_allocated()
4. [ ] Run D3 component test on both (if applicable)
5. [ ] Measure inference speed (tokens/second)

### Phase 5: Analysis
1. [ ] Compute loss ratio: GF-MH loss / GPT-2 loss
2. [ ] Apply decision thresholds
3. [ ] Document findings

---

## 5. Success Criteria (from V5_GATING.md)

### Primary Criterion: Loss Ratio
| Ratio | Interpretation | Action |
|-------|----------------|--------|
| ≤ 0.95 | GF-MH significantly better | Celebrate, document |
| 0.95-1.05 | Equivalent | Proceed, compare secondary |
| 1.05-1.20 | GF-MH slightly worse | Proceed if secondary metrics advantage |
| 1.20-1.30 | GF-MH notably worse | Investigate, may proceed with caution |
| > 1.30 | GF-MH significantly worse | STOP, debug architecture |

### Secondary Criteria
| Metric | "Promising" | "Excellent" |
|--------|-------------|-------------|
| Training Speed | 20% faster | 50% faster |
| Memory Usage | 20% less | 50% less |
| Statefulness | 25% better | 50% better |

### Decision Rule
- **PROCEED** if: Loss ratio ≤ 1.20 AND at least one secondary metric is "Promising"
- **STOP** if: Loss ratio > 1.30 OR all secondary metrics worse

---

## 6. Expected Results

Based on Observation 16 (architectural analysis):
- GF-MH has higher per-step compute (parallel RWKV+Mamba)
- GF-MH may be slower per step but potentially better sample efficiency
- Statefulness should favor GF-MH (inherent state vs attention window)

**Prediction:** Loss ratio 0.95-1.10, with statefulness advantage.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Param count mismatch | Verify before training |
| Different effective learning | Use same lr, but monitor loss curves |
| Hardware variance | Run sequentially on same GPU |
| Random variance | Use fixed seed, could add multiple runs |

---

## 8. Output Artifacts

After experiment completion:
- [ ] `logs/exp001_gfmh/` — GF-MH training logs
- [ ] `logs/exp001_gpt2/` — GPT-2 training logs
- [ ] `checkpoints/exp001_gfmh_step500.pt` — GF-MH checkpoint
- [ ] `checkpoints/exp001_gpt2_step500.pt` — GPT-2 checkpoint
- [ ] `data/batch_order.npy` — Shared batch order
- [ ] Observation 17 in V4_FUSION_MODELS.md

---

## 9. Sign-off

- [ ] Protocol reviewed
- [ ] Controls verified
- [ ] Ready to proceed

**Approved by:** _______________  
**Date:** _______________
