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
