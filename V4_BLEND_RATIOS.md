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
| 1 | GF-MH | 0.3 | 30:70 (Mamba) | ✅ Already done — baseline |
| 2 | GF-RH | 0.7 | 70:30 (RWKV) | Does it stay RWKV-heavy? |
| 3 | HGF | 0.5 | 50:50 | ✅ Already done — baseline |
| 4 | HGF-MH | 0.3 | 30:70 (Mamba) | Per-dim gates + Mamba bias |
| 5 | HGF-RH | 0.7 | 70:30 (RWKV) | Per-dim gates + RWKV bias |
| 6 | HY | fixed 0.7:0.3 | 70:30 (RWKV) | ✅ Already done — compare to gated |

### Key Question for Each Experiment

After 1K steps, measure:
1. **Final gate value** (or gain ratio for HY)
2. **R/M gradient ratio** 
3. **Val loss**

If GF-RH (started 70:30 RWKV) drifts toward equal or Mamba-heavy → **Signal dominance**
If GF-RH stays 70:30 RWKV → **Architectural persistence**

---

## Interpreting Results

### Scenario A: Signal Dominance (RWKV produces easier gradients)

```
GF-MH (start 30:70) → Final gate ~0.7 (learned RWKV preference)
GF-RH (start 70:30) → Final gate ~0.7 (stayed RWKV)
```
**Meaning:** Model naturally gravitates to RWKV regardless of init.
**Action:** Need explicit Mamba encouragement (loss penalties, separate LRs)

### Scenario B: Architectural Persistence (init matters)

```
GF-MH (start 30:70) → Final gate ~0.4 (drifted toward balance)
GF-RH (start 70:30) → Final gate ~0.6 (drifted toward balance)
```
**Meaning:** Model converges to optimal blend from either direction.
**Action:** 50/50 init is fine, balance is learned

### Scenario C: Asymmetric (one direction stable)

```
GF-MH (start 30:70) → Final gate ~0.7 (RWKV won)
GF-RH (start 70:30) → Final gate ~0.7 (RWKV stayed)
```
**Meaning:** RWKV "stickier" — Mamba needs head start.
**Action:** Use Mamba-heavy init + balance loss

---

## Commands for Phase 3.7 Testing

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

### Phase 3.7 Results (To Be Filled)

**Test Date:** ____  
**Conditions:** batch=32, seq_len=64, 1000 steps, Shakespeare char-level

| Model | gate_init | Final Gate | R/M Ratio | Val Loss | Val PPL |
|-------|-----------|------------|-----------|----------|---------|
| GF-MH | 0.3 | — | 0.10 | 1.59 | 4.90 | *(Phase 3.6)* |
| GF-RH | 0.7 | — | — | — | — |
| HGF | 0.5 | — | 0.21 | 1.69 | 5.41 | *(Phase 3.6)* |
| HGF-MH | 0.3 | — | — | — | — |
| HGF-RH | 0.7 | — | — | — | — |

**Analysis:**

- GF-RH drift: (init 0.7 → final ?)
- HGF-RH drift: (init 0.7 → final ?)
- Conclusion: [Architectural / Signal / Asymmetric]

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
