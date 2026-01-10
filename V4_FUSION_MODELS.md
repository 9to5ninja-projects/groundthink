# V4 Fusion Models — Technical Reference

**Purpose:** Single source of truth for all fusion variants  
**Updated:** 2026-01-10  
**Audience:** Developers, future agents, decision-makers

---

## Core Architecture (All Variants Share This)

Every V4 model uses **parallel hybrid blocks**. Both RWKV-6 and Mamba-2 process the same input simultaneously:

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT x [batch, seq, hidden=128]                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  norm_x = RMSNorm(x)                                        │
│                                                             │
│  ┌─────────────┐     ┌─────────────┐                        │
│  │   RWKV-6    │     │   Mamba-2   │   ← PARALLEL           │
│  │  (smooth    │     │  (selective │                        │
│  │   decay)    │     │   gating)   │                        │
│  └──────┬──────┘     └──────┬──────┘                        │
│         │                   │                               │
│         ▼                   ▼                               │
│      out_rwkv           out_mamba                           │
│         │                   │                               │
│         └─────────┬─────────┘                               │
│                   ▼                                         │
│           ┌──────────────┐                                  │
│           │ FUSION (var) │  ← THIS IS THE DIFFERENCE        │
│           └──────┬───────┘                                  │
│                  ▼                                          │
│              fused                                          │
│                  │                                          │
│  output = x + fused + FFN(RMSNorm(x + fused))              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** The fusion method determines HOW the model blends RWKV and Mamba signals. This happens at every block (8 blocks in small model).

---

## Fusion Variants

### 1. HY — Hybrid Per-Channel Gains

**File:** `models/hybrid_v4.py`  
**Fusion params:** 256 (at hidden=128)

```python
# Per-dimension learnable gains
rwkv_gain = Parameter(ones(128) * 0.7)   # Init: favor RWKV
mamba_gain = Parameter(ones(128) * 0.3)  # Init: less Mamba

# Fusion (after output normalization)
out_rwkv = RMSNorm(rwkv6(norm_x))
out_mamba = RMSNorm(mamba2(norm_x))
fused = rwkv_gain * out_rwkv + mamba_gain * out_mamba
```

**Granularity:** Per-dimension, FIXED across positions  
**What it learns:** "Dimension 47 should be 80% RWKV, dimension 98 should be 60% Mamba"  
**Limitation:** Same blend for all token positions in sequence

---

### 2. GF — Gated Fusion

**File:** `models/hybrid_v4_GF.py`  
**Fusion params:** 257 (256 weights + 1 bias)

```python
# Gate projection: [batch, seq, 256] → [batch, seq, 1]
gate_proj = Linear(hidden * 2, 1, bias=True)

# Fusion
combined = cat([out_rwkv, out_mamba], dim=-1)  # [B, S, 256]
gate = sigmoid(gate_proj(combined))             # [B, S, 1] scalar per position
fused = gate * out_rwkv + (1-gate) * out_mamba
```

**Granularity:** Per-position, SAME across dimensions  
**What it learns:** "Position 5 should be 70% RWKV, position 12 should be 30% RWKV"  
**Limitation:** All 128 dimensions get same blend at each position

---

### 3. GF-MH — Gated Fusion, Mamba-Heavy ★ PHASE 2 WINNER

**File:** `models/hybrid_v4_ratio.py`  
**Fusion params:** 257

Same as GF, but gate initialized to favor Mamba:

```python
# Init gate bias so sigmoid(bias) ≈ 0.3 (30% RWKV, 70% Mamba)
gate_init = 0.3
init_bias = log(gate_init / (1 - gate_init))  # ≈ -0.847
gate_proj.bias.fill_(init_bias)
gate_proj.weight.zero_()  # Pure bias at start
```

**Why it won:** Mamba's selective gating handles instruction-following better; RWKV provides smooth memory. Starting Mamba-heavy lets model build on that strength.

---

### 4. CP — Concatenate + Project

**File:** `models/hybrid_v4_CP.py`  
**Fusion params:** 32,896 (256×128 + 128 bias = 32,896)

```python
# Full learned projection
fusion_proj = Linear(hidden * 2, hidden, bias=False)

# Fusion
combined = cat([out_rwkv, out_mamba], dim=-1)  # [B, S, 256]
fused = fusion_proj(combined)                   # [B, S, 128]
```

**Granularity:** Fully learned — can express ANY linear combination  
**What it learns:** Arbitrary mixing matrix, not constrained to interpolation  
**Tradeoff:** Most params but also most expressive

---

### 5. WS — Weighted Sum

**File:** `models/hybrid_v4_WS.py`  
**Fusion params:** 1

```python
# Single scalar weight
w = Parameter(tensor(0.5))

# Fusion
fused = w * out_rwkv + (1-w) * out_mamba
```

**Granularity:** Global scalar, same everywhere  
**Limitation:** Too simple — can't adapt to context

---

### 6. RF — Residual Fusion

**File:** `models/hybrid_v4_RF.py`  
**Fusion params:** 1

```python
# Mamba adds to RWKV as residual
alpha = Parameter(tensor(0.3))

# Fusion
fused = out_rwkv + alpha * out_mamba
```

**Granularity:** Global scalar residual  
**Limitation:** Asymmetric — RWKV is "base", Mamba is "adjustment"

---

## Comparison Matrix

| Variant | Params | Position-Adaptive | Dimension-Adaptive | Phase 2 Val Loss |
|---------|--------|-------------------|-------------------|------------------|
| **GF-MH** | 257 | ✅ Yes | ❌ No | **1.670** ★ |
| **CP** | 32,896 | ✅ Yes | ✅ Yes (implicit) | 1.692 |
| **GF** | 257 | ✅ Yes | ❌ No | 1.689 |
| **HY** | 256 | ❌ No | ✅ Yes | 1.760 |
| **WS** | 1 | ❌ No | ❌ No | 1.819 |
| **RF** | 1 | ❌ No | ❌ No | 1.948 |

---

## Complete Results: Phase 3.6 + 3.7 (2026-01-10)

**Test conditions:** batch=32, seq_len=64, 1000 steps, Shakespeare char-level

| Variant | Fusion Type | gate_init | Val Loss | Val PPL | R/M Ratio | Status |
|---------|-------------|-----------|----------|---------|-----------|--------|
| **GF-MH** | Per-position | 0.3 | **1.59** | **4.90** | 0.10 | ⚠️ Worst balance |
| **GF** | Per-position | 0.5 | 1.61 | 5.00 | 0.12 | ⚠️ RWKV dominant |
| **GF-RH** | Per-position | 0.7 | 1.64 | 5.14 | 0.14 | ⚠️ RWKV dominant |
| **CP** | Projection | — | 1.61 | 4.98 | 0.19 | ⚠️ RWKV dominant |
| **HGF** | Per-pos+dim | 0.5 | 1.69 | 5.41 | 0.21 | ⚠️ RWKV dominant |
| **HGF-MH** | Per-pos+dim | 0.3 | 1.69 | 5.40 | 0.24 | ⚠️ Best gated balance |
| **HGF-RH** | Per-pos+dim | 0.7 | 1.70 | 5.46 | 0.25 | ⚠️ RWKV dominant |
| **HY** | Fixed per-dim | — | 1.69 | 5.42 | **0.45** | ✅ Best balance |

---

## Key Findings (Phase 3.7 Analysis)

### 1. Signal Dominance Confirmed
ALL gated variants converge to RWKV-dominant regardless of initial gate bias.
- GF-MH (started Mamba-heavy) → R/M 0.10 (RWKV dominant)
- GF-RH (started RWKV-heavy) → R/M 0.14 (RWKV dominant)
- **Root cause:** RWKV produces smoother gradients, easier to optimize

### 2. Per-Dimension Gating Preserves Mamba Better
| Fusion Type | Avg R/M | Why |
|-------------|---------|-----|
| Per-position (GF) | 0.12 | One gate for all dims → total collapse |
| Per-pos+dim (HGF) | 0.23 | 128 gates → some dims keep Mamba |
| Fixed per-dim (HY) | 0.45 | Can't drift |

### 3. Loss-Balance Pareto Frontier
```
Best Loss ←─────────────────────────────→ Best Balance
  GF-MH (1.59, R/M 0.10) ... HGF-MH (1.69, R/M 0.24) ... HY (1.69, R/M 0.45)
```

### 4. Recommendations by Use Case

| Goal | Model | Loss | R/M | Why |
|------|-------|------|-----|-----|
| Lowest loss | GF-MH | 1.59 | 0.10 | Best performance, accept imbalance |
| Best balance | HY | 1.69 | 0.45 | Fixed gains can't drift |
| Balance + adaptivity | HGF-MH | 1.69 | 0.24 | Best gated variant for balance |
| Middle ground | CP | 1.61 | 0.19 | Good loss, moderate imbalance |

---

## Observations & Patterns (Phase 3.6-3.7)

This section documents data-driven observations from fusion variant experiments. These patterns inform future design decisions.

### Observation 1: Per-Dimension Gating Preserves Mamba Better

| Fusion Type | Variants | R/M Range | Avg R/M |
|-------------|----------|-----------|---------|
| Per-position (GF) | GF, GF-MH, GF-RH | 0.10-0.14 | 0.12 |
| Per-pos+dim (HGF) | HGF, HGF-MH, HGF-RH | 0.21-0.25 | 0.23 |
| Fixed per-dim (HY) | HY | 0.45 | 0.45 |

**Pattern:** HGF achieves ~2x better balance than GF with same expressivity.

**Why?** GF applies ONE gate value to all 128 dimensions. If gate→RWKV, ALL dimensions lose Mamba simultaneously. HGF has 128 independent gates per position — some dimensions can stay Mamba-heavy even when others drift.

**Implication:** Per-dimension control is critical for component balance in hybrid architectures.

---

### Observation 2: Loss and Balance Are Inversely Correlated

```
Loss-Balance Pareto Frontier:

Best Loss                                           Best Balance
    ↓                                                     ↓
  GF-MH ──── GF ──── CP ──── HGF ──── HGF-MH ──── HY
  (1.59)    (1.61)  (1.61)  (1.69)    (1.69)    (1.69)
  R/M=0.10  R/M=0.12 R/M=0.19 R/M=0.21 R/M=0.24  R/M=0.45
```

**Pattern:** You cannot optimize for both loss AND balance without explicit intervention.

**Why?** The optimizer takes the "path of least resistance." RWKV produces smoother gradients, so the gate learns to favor RWKV to minimize loss faster. This is signal dominance.

**Implication:** Achieving balance requires either:
- Fixed gains (HY) that cannot drift
- Explicit balance regularization in the loss function
- Component-specific learning rates

---

### Observation 3: More Fusion Params ≠ Better Balance

| Model | Fusion Params | Constraint | R/M | Val Loss |
|-------|---------------|------------|-----|----------|
| WS | 1 | None | — | 1.82 |
| RF | 1 | Asymmetric | — | 1.95 |
| HY | 256 | Fixed gains | 0.45 | 1.69 |
| GF | 257 | Per-position sigmoid | 0.12 | 1.61 |
| CP | 32,896 | Free linear projection | 0.19 | 1.61 |
| HGF | 32,896 | Per-dim sigmoid | 0.21-0.25 | 1.69 |

**Pattern:** Constraint semantics matter more than param count.

- CP (33K params, free projection): R/M 0.19
- HGF (33K params, sigmoid [0,1]): R/M 0.21-0.25

**Why?** CP can learn to "cancel out" Mamba entirely with negative weights. HGF's sigmoid constraint forces true interpolation — each dimension MUST be a blend of RWKV and Mamba.

**Implication:** Architectural constraints (sigmoid, gain clipping) can enforce balance better than raw expressivity.

---

### Observation 4: Gate Initialization Doesn't Matter (Signal Dominance)

| Model | gate_init | Expected | Actual R/M | Drift Direction |
|-------|-----------|----------|------------|-----------------|
| GF-MH | 0.3 (Mamba) | Mamba-heavy | 0.10 | → RWKV |
| GF | 0.5 (balanced) | Balanced | 0.12 | → RWKV |
| GF-RH | 0.7 (RWKV) | RWKV-heavy | 0.14 | stayed RWKV |
| HGF-MH | 0.3 (Mamba) | Mamba-heavy | 0.24 | → RWKV |
| HGF | 0.5 (balanced) | Balanced | 0.21 | → RWKV |
| HGF-RH | 0.7 (RWKV) | RWKV-heavy | 0.25 | stayed RWKV |

**Pattern:** ALL gated variants converge to RWKV-dominant regardless of initial bias.

**Why?** RWKV's smooth decay produces gradients that are easier for the optimizer to follow. Mamba's selective gating creates spikier, harder-to-optimize signals.

**Implication:** `gate_init` is not a viable hyperparameter for balance tuning. The optimizer will override any initial bias.

---

### Observation 5: HGF is the Best Gated Variant for Balance

| Model | Position-Adaptive | Dim-Adaptive | R/M | Loss |
|-------|-------------------|--------------|-----|------|
| GF-MH | ✅ | ❌ | 0.10 | 1.59 |
| CP | ✅ | ✅ (free) | 0.19 | 1.61 |
| HGF-MH | ✅ | ✅ (constrained) | 0.24 | 1.69 |
| HY | ❌ | ✅ | 0.45 | 1.69 |

**Pattern:** HGF-MH achieves same loss as HY but with position-adaptive gating.

**Recommendation:** If you want gated fusion with reasonable balance, use **HGF-MH** over GF-MH.

---

### Observation 6: Activation Variance Reveals Component Health

During training, we track `RWKV var` and `Mamba var`:

| Model | RWKV var | Mamba var | Ratio | Health |
|-------|----------|-----------|-------|--------|
| GF-MH | 23.6 | 0.12 | 197:1 | ⚠️ Mamba collapsed |
| HGF-MH | 7.3 | 0.15 | 49:1 | ⚠️ Imbalanced but alive |
| HY | ~4.0 | ~0.16 | 25:1 | ✅ Best gated |

**Pattern:** RWKV activation variance is consistently 25-200x higher than Mamba's.

**Why?** RWKV's smooth decay preserves signal magnitude. Mamba's selective gating aggressively attenuates most positions.

**Implication:** Activation variance ratio is a diagnostic metric. Ratios >100:1 indicate Mamba is effectively dead.

---

### Observation 7: Training Dynamics — R/M Ratio Drifts Over Time

Typical trajectory for gated variants:

```
Steps:   0 ────── 200 ────── 500 ────── 1000
R/M:    0.7-1.2    0.4-0.6    0.2-0.3    0.10-0.25
        ↑ random   ↑ drifting  ↑ settling ↑ final
```

**Pattern:** R/M starts near 1.0 and drifts toward RWKV-dominant during training.

**When?** The drift accelerates during warmup (steps 0-500) when learning rate is ramping up.

**Implication:** Gate freezing during warmup (Task 37) may prevent early RWKV lock-in.

---

### Summary Table: Fusion Variant Characteristics

| Variant | Loss | R/M | Position-Adapt | Dim-Adapt | Params | Best For |
|---------|------|-----|----------------|-----------|--------|----------|
| GF-MH | ★★★★★ | ★ | ✅ | ❌ | 257 | Pure loss optimization |
| GF | ★★★★ | ★ | ✅ | ❌ | 257 | Simple gating baseline |
| CP | ★★★★ | ★★ | ✅ | ✅ | 33K | Maximum expressivity |
| HGF-MH | ★★★ | ★★★ | ✅ | ✅ | 33K | Balance + adaptivity |
| HGF | ★★★ | ★★ | ✅ | ✅ | 33K | Middle ground |
| HY | ★★★ | ★★★★★ | ❌ | ✅ | 256 | Best component balance |
| WS | ★★ | — | ❌ | ❌ | 1 | Not recommended |
| RF | ★ | — | ❌ | ❌ | 1 | Not recommended |

---

## The Missing Piece: HGF (Hybrid-Gated Fusion)

**Implemented:** Combines HY's per-dimension control with GF's per-position adaptivity.

```python
# Per-position, per-dimension gates
gate_proj = Linear(hidden * 2, hidden)  # [B, S, 256] → [B, S, 128]

# Fusion
combined = cat([out_rwkv, out_mamba], dim=-1)
gate = sigmoid(gate_proj(combined))      # [B, S, 128] — gate per dim per pos
fused = gate * out_rwkv + (1-gate) * out_mamba  # Elementwise
```

**Granularity:** Per-position AND per-dimension  
**Fusion params:** 32,896 (same as CP, but with interpolation semantics)  
**What it learns:** "Position 5, dimension 47 should be 80% RWKV; position 5, dimension 98 should be 40% RWKV"

**Hypothesis:** More control over the "shape" of the learning field. The gate explicitly learns WHERE (position) and WHAT (dimension) to blend.

---

## Recommended Top 3 for Extended Testing

1. **GF-MH** — Current winner, baseline for comparison
2. **CP** — Most expressive, see if extra params help at scale
3. **HGF** — New variant, test the per-position-per-dimension hypothesis

---

## Usage

```python
from models import get_model, list_models

list_models(show=True)  # See all available

# Load specific variant
model = get_model('GF-MH')   # Phase 2 winner
model = get_model('CP')      # Concat+Project
model = get_model('HGF')     # New hybrid-gated (after implementation)
```
