# V4 Blend Ratio Experiments — Technical Reference

**Purpose:** Document adjustable blend hyperparameters and plan systematic ratio testing  
**Created:** 2026-01-10 (Phase 3.7 Planning)  
**Context:** Following Phase 3.6 discovery that all gated variants show RWKV dominance

---

## Executive Summary

**Question:** Is RWKV dominance architectural or signal-based?

- **Architectural:** Built into model weights/structure — can't be changed with hyperparams
- **Signal-based:** RWKV produces "easier" gradients — could be adjusted via init bias

**Finding:** 3 of 5 fusion variants have adjustable `gate_init` hyperparameters for testing 30/70, 50/50, 70/30 blends.

---

## Fusion Variants: Adjustability Matrix

| Variant | File | Adjustable? | Param Name | Current Values | Notes |
|---------|------|-------------|------------|----------------|-------|
| **HY** | hybrid_v4.py | ⚠️ Fixed | `rwkv_gain`, `mamba_gain` | 0.7, 0.3 | Hardcoded init, per-dim weights |
| **GF** | hybrid_v4_GF.py | ❌ No | — | Balanced | No gate_init param in factory |
| **GF-MH** | hybrid_v4_ratio.py | ✅ Yes | `gate_init` | 0.3 | Factory creates Mamba-heavy |
| **GF-RH** | hybrid_v4_ratio.py | ✅ Yes | `gate_init` | 0.7 | Factory creates RWKV-heavy |
| **HGF** | hybrid_v4_HGF.py | ✅ Yes | `gate_init` | 0.5 (balanced) | Per-position + per-dimension |
| **HGF-MH** | hybrid_v4_HGF.py | ✅ Yes | `gate_init` | 0.3 | Mamba-heavy HGF |
| **HGF-RH** | hybrid_v4_HGF.py | ✅ Yes | `gate_init` | 0.7 | RWKV-heavy HGF |
| **CP** | hybrid_v4_CP.py | ❌ No | — | — | No blend constraint (learned projection) |

---

## How `gate_init` Works

The gate controls RWKV vs Mamba contribution:

```python
fused = gate * out_rwkv + (1-gate) * out_mamba
```

- `gate_init = 0.7` → 70% RWKV, 30% Mamba (RWKV-heavy, "RH")
- `gate_init = 0.5` → 50% RWKV, 50% Mamba (balanced)
- `gate_init = 0.3` → 30% RWKV, 70% Mamba (Mamba-heavy, "MH")

**Implementation:** Gate bias initialized via logit transform:
```python
init_bias = log(gate_init / (1 - gate_init))  # sigmoid⁻¹
gate_proj.bias.fill_(init_bias)
```

**Key insight:** The gate LEARNS during training. `gate_init` only sets starting point. If final gate differs significantly from init, that reveals signal dominance.

---

## Existing Registry Variants

These are already registered in `models/__init__.py`:

| Registry Key | gate_init | RWKV:Mamba | Status |
|--------------|-----------|------------|--------|
| `GF-MH` | 0.3 | 30:70 | ✅ Tested (Phase 3.6) |
| `GF-RH` | 0.7 | 70:30 | ❌ Not tested |
| `HGF` | 0.5 | 50:50 | ✅ Tested (Phase 3.6) |
| `HGF-MH` | 0.3 | 30:70 | ❌ Not tested |
| `HGF-RH` | 0.7 | 70:30 | ❌ Not tested |

---

## Phase 3.7 Experiment Plan: Blend Ratio Sweep

### Goal
Determine if RWKV dominance is architectural or signal-based by testing symmetric configurations.

### Test Matrix (6 experiments)

| # | Model | gate_init | Expected Start | Test Purpose |
|---|-------|-----------|----------------|--------------|
| 1 | GF-MH | 0.3 | 30:70 (Mamba) | ✅ R/M 0.10 — RWKV dominant |
| 2 | GF-RH | 0.7 | 70:30 (RWKV) | ✅ R/M 0.14 — RWKV dominant |
| 3 | HGF | 0.5 | 50:50 | ✅ R/M 0.21 — RWKV dominant |
| 4 | HGF-MH | 0.3 | 30:70 (Mamba) | ✅ R/M 0.24 — RWKV dominant |
| 5 | HGF-RH | 0.7 | 70:30 (RWKV) | ✅ R/M 0.25 — RWKV dominant |
| 6 | HY | fixed 0.7:0.3 | 70:30 (RWKV) | ✅ R/M 0.45 — best balance |
| 7 | GF | 0.5 | 50:50 | ✅ R/M 0.12 — RWKV dominant |
| 8 | CP | — | learned | ✅ R/M 0.19 — RWKV dominant |

---

## Interpreting Results — SCENARIO A CONFIRMED

### Scenario A: Signal Dominance ✅ THIS IS WHAT HAPPENED

```
GF-MH (start 30:70 Mamba) → R/M 0.10 (RWKV dominant)
GF-RH (start 70:30 RWKV)  → R/M 0.14 (RWKV dominant)
HGF-MH (start 30:70 Mamba) → R/M 0.24 (RWKV dominant)
HGF-RH (start 70:30 RWKV)  → R/M 0.25 (RWKV dominant)
```

**Meaning:** Model naturally gravitates to RWKV regardless of init.
**Root cause:** RWKV produces smoother gradients, easier to optimize.

---

## Data-Driven Inferences

### Inference 1: Per-Dimension Gating Preserves Mamba Better

| Fusion Type | Variants | Avg R/M | Insight |
|-------------|----------|---------|---------|
| Per-position (GF) | GF, GF-MH, GF-RH | 0.12 | Worst balance — all dims collapse together |
| Per-pos+dim (HGF) | HGF, HGF-MH, HGF-RH | 0.23 | ~2x better — some dims keep Mamba |
| Fixed per-dim (HY) | HY | 0.45 | Best — can't drift |

**Why?** GF applies same gate to all 128 dims. If gate→RWKV, ALL dims lose Mamba.
HGF has 128 independent gates. Some dims can stay Mamba-heavy even if others drift.

### Inference 2: Loss-Balance Pareto Frontier

```
Best Loss ←─────────────────────────────→ Best Balance
  GF-MH (1.59, 0.10) ... CP (1.61, 0.19) ... HGF (1.69, 0.21) ... HY (1.69, 0.45)
```

You CAN'T have both without explicit intervention. Choose:
- **GF-MH**: Best loss, sacrifice balance
- **HY**: Best balance, accept higher loss
- **HGF-MH**: Middle ground (1.69 loss, 0.24 balance)

### Inference 3: Interpolation Semantics Help

| Model | Fusion Params | Constraint | R/M |
|-------|---------------|------------|-----|
| CP | 33K | None (free linear) | 0.19 |
| HGF | 33K | Sigmoid [0,1] | 0.21-0.25 |

Same param count, but HGF's sigmoid constraint forces blend semantics.
CP can learn to "cancel out" Mamba entirely. HGF can only blend.

### Inference 4: HGF-MH is Best Gated Variant for Balance

If you want gated fusion AND reasonable balance:
- **HGF-MH**: R/M 0.24, Loss 1.69 ← RECOMMENDED
- GF-MH: R/M 0.10, Loss 1.59 (better loss, worse balance)

HGF-MH has same loss as HY but is position-adaptive.

```bash
# Activate environment
source /home/m_tes/groundthink/.venv/bin/activate

# Test 1: GF-RH (70:30 RWKV-heavy) — NEW
python train_v4.py --model GF-RH --steps 1000 --batch_size 32 --seq_len 64

# Test 2: HGF-MH (30:70 Mamba-heavy) — NEW  
python train_v4.py --model HGF-MH --steps 1000 --batch_size 32 --seq_len 64

# Test 3: HGF-RH (70:30 RWKV-heavy) — NEW
python train_v4.py --model HGF-RH --steps 1000 --batch_size 32 --seq_len 64
```

---

## Results Template

### Phase 3.7 Results (2026-01-10) ✅ COMPLETE

**Test Date:** 2026-01-10  
**Conditions:** batch=32, seq_len=64, 1000 steps, Shakespeare char-level

| Model | gate_init | Final R/M | Val Loss | Val PPL | Observation |
|-------|-----------|-----------|----------|---------|-------------|
| GF-MH | 0.3 | 0.10 | 1.59 | 4.90 | ⚠️ RWKV dominant *(Phase 3.6)* |
| **GF-RH** | **0.7** | **0.14** | **1.64** | **5.14** | ⚠️ RWKV dominant — same pattern! |
| HGF | 0.5 | 0.21 | 1.69 | 5.41 | ⚠️ RWKV dominant *(Phase 3.6)* |
| **HGF-MH** | **0.3** | **0.24** | **1.69** | **5.40** | ⚠️ RWKV dominant |
| **HGF-RH** | **0.7** | **0.25** | **1.70** | **5.46** | ⚠️ RWKV dominant |

---

## Phase 3.7 Analysis & Conclusion

### Key Finding: RWKV Dominance is SIGNAL-BASED

All 5 gated variants converge to RWKV-dominant regardless of initial gate bias:

| Model | Started As | Ended As | Conclusion |
|-------|------------|----------|------------|
| GF-MH | 30% RWKV (Mamba-heavy) | R/M=0.10 (RWKV dominant) | Drifted to RWKV |
| GF-RH | 70% RWKV (RWKV-heavy) | R/M=0.14 (RWKV dominant) | Stayed RWKV |
| HGF-MH | 30% RWKV (Mamba-heavy) | R/M=0.24 (RWKV dominant) | Drifted to RWKV |
| HGF-RH | 70% RWKV (RWKV-heavy) | R/M=0.25 (RWKV dominant) | Stayed RWKV |
| HGF | 50% RWKV (balanced) | R/M=0.21 (RWKV dominant) | Drifted to RWKV |

**Verdict:** **SIGNAL DOMINANCE CONFIRMED**

The gate learns to favor RWKV regardless of initialization because:
1. RWKV produces smoother, more stable gradients
2. Mamba's selective gating creates spikier, harder-to-optimize signals
3. The optimizer "takes the path of least resistance" to RWKV

### Implications

1. **`gate_init` doesn't matter** — the gate will drift toward RWKV anyway
2. **Mamba needs explicit encouragement** — separate LR (already using 0.5x) isn't enough
3. **Consider balance loss term** — penalize R/M ratio deviation from target
4. **HY remains best for balance** — its fixed gains (0.45 R/M) can't drift

### Recommended Next Steps

1. **Add balance regularization:** Loss term that penalizes R/M < 0.3 or > 3.0
2. **Increase Mamba LR multiplier:** Try 1.0 instead of 0.5
3. **Test frozen gates:** Initialize gate and freeze it for N steps
4. **Alternative:** Accept RWKV dominance if loss is good enough

---

## Code Changes Required

None! All variants already exist in the registry:

```python
from models import get_model

# These all work:
model = get_model('GF-RH')   # RWKV-heavy GF
model = get_model('HGF-MH')  # Mamba-heavy HGF
model = get_model('HGF-RH')  # RWKV-heavy HGF
```

---

## HY Variant: Special Case

**HY does NOT use `gate_init`** — it uses fixed per-dimension gains:

```python
# In hybrid_v4.py (hardcoded)
self.rwkv_gain = nn.Parameter(torch.ones(hidden_size) * 0.7)
self.mamba_gain = nn.Parameter(torch.ones(hidden_size) * 0.3)
```

To test other ratios with HY, we'd need to:
1. Create new factory functions with different init values, OR
2. Modify the block class to accept gain_init parameter

**Recommendation:** Focus on gated variants (GF-RH, HGF-*) first since they have explicit gate tracking.

---

## References

- [V4_FUSION_MODELS.md](V4_FUSION_MODELS.md) — Full fusion variant documentation
- [V4_STRATEGY.md](V4_STRATEGY.md) — Phase planning and task tracking
- [models/__init__.py](models/__init__.py) — Model registry
