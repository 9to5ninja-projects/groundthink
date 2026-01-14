# Task 0.1 GRU Arbiter Exploration - Findings

**Date:** 2025-01-13  
**Version:** 0.5.1.1  
**Notebook:** `notebooks/task_0_1_exploration.ipynb`

---

## Experiment Results

| Exp | Name | Key Metric | Result |
|-----|------|------------|--------|
| 1 | Constant Input | α = 0.499 ± 0.015 | ✓ PASS - No init bias |
| 2 | Synthetic Divergence | α = 0.493 ± 0.020 | ⚠ FLAT - No variance response |
| 3-4 | Real Outputs | α = 0.502 ± 0.0006 | ⚠ FLAT - No response to 4x norm diff |
| 6 | Hidden State | Norm 0.36 → 0.75 | ✓ PASS - GRU accumulates info |
| 7 | Trainability | Loss -94% | ✓ PASS - Arbiter learns |

---

## Detailed Observations

### Exp 1: Constant Input
- α_rwkv: 0.499 ± 0.015
- α_mamba: 0.501 ± 0.015
- Hidden state norm: 3.97
- **Interpretation:** Balanced initialization, no preference when inputs identical

### Exp 2: Synthetic Divergence
- Amplifier norm: 12.26 → 33.53 (3x growth)
- Damper norm: 11.20 → 3.24 (3x decay)
- α_rwkv: 0.493 ± 0.020
- **Interpretation:** Untrained arbiter doesn't respond to massive variance differences

### Exp 3-4: Real RWKV & Mamba Outputs
- RWKV norm: 0.459 ± 0.072
- Mamba norm: 0.118 ± 0.013 (4x smaller)
- α_rwkv: 0.502 ± 0.0006
- **Interpretation:** Even with real architectural differences, α stays flat

### Exp 6: Hidden State Analysis
- Norm evolution: 0.36 → 0.75 (doubles)
- Mean: 0.755, Std: 0.060
- **Interpretation:** GRU IS accumulating information, problem is downstream

### Exp 7: Trainability
- Initial loss: ~0.032
- Final loss: 0.0018
- Reduction: **94%**
- Pred vs Target at end: 0.412 vs 0.454
- **Interpretation:** Arbiter CAN learn meaningful gating

---

## Diagnosis

| Component | Status | Notes |
|-----------|--------|-------|
| GRU recurrence | ✓ Working | Hidden state evolves meaningfully |
| weight_proj init | ⚠ Too weak | std=0.01 causes flat softmax output |
| Trainability | ✓ Confirmed | 94% loss reduction proves learning |
| Architecture | ✓ Viable | No fundamental issues |

**Root Cause:** `weight_proj` initialized with std=0.01, producing near-zero logits → softmax saturates to [0.5, 0.5].

**Fix Options:**
1. Warmer init (std=0.1) for faster convergence
2. Trust training to learn it (works, just slower)
3. Consider GLU/minGRU for efficiency (not necessity)

---

## Conclusions

1. **GRU Arbiter architecture is VIABLE**
2. Untrained behavior (flat α) is an init problem, not design flaw
3. Hidden state carries temporal information
4. Training successfully teaches gating behavior
5. GLU/minGRU worth exploring for **efficiency**, not because GRU failed

---

## Next Steps

**Decision Point:** Choose arbiter mechanism before Task 0.2

| Option | Pros | Cons |
|--------|------|------|
| Keep GRU | Proven works, already implemented | More params, slower |
| minGRU | Simpler, fewer params | Need to implement |
| GLU | No recurrence, fastest | May lose temporal context |

**Recommendation:** Review minGRU/GLU docs, make architecture decision, then proceed to Task 0.2 (Residual Connections).
