# GroundThink V4 Design Document

**Created:** 2026-01-08  
**Updated:** 2026-01-09 (Phase 2 Complete)
**Status:** Parallel Block Architecture - IMPLEMENTED (hybrid_v4_ratio.py - GF-MH Winner)  
**Author:** User (not agents)
**Current Implementation:** 8-block parallel hybrid with Gated Fusion, 3.5M params

---

## 🐧 ENVIRONMENT: NATIVE LINUX (UBUNTU)

**We are working in native Linux via VS Code Remote-SSH.**

- All file paths use Linux syntax: `/home/m_tes/groundthink/...`
- Terminal commands are bash/Linux (not Windows/PowerShell)
- Python venv at `.venv/` (use `source .venv/bin/activate`)
- CUDA kernels (causal-conv1d, mamba-ssm) are installed and working

**Do NOT use Windows paths or syntax in any commands or scripts.**

---

## ⛔ CRITICAL: READ BEFORE DOING ANYTHING

**V3 was scrapped because agents ignored the architecture spec and built RWKV-7 instead of RWKV-6 + Mamba-2.**

This document defines what V4 MUST be. Do not substitute. Do not "improve." Do not default to whatever is in your training data.

If you are an agent reading this:
1. **RWKV-6** means RWKV-6, not RWKV-7
2. **Mamba-2** means actual SSM with A/B/C/D matrices
3. **Hybrid** means TWO SEPARATE COMPONENTS, not one component doing both
4. If you don't know how to implement RWKV-6 or Mamba-2, **ASK** - do not substitute

---

## 🔧 RUNTIME REQUIREMENTS (Windows + FLA/Triton)

**Every script must include these or it will crash:**

```python
# 1. FIX OPENMP CONFLICT - must be BEFORE any torch/numpy imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

# 2. ALL TENSORS ON CUDA - FLA uses Triton kernels, CPU tensors crash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = torch.randint(0, vocab, (batch, seq), device=device)  # device= required!
```

**Errors you'll see if you forget:**
- `OMP: Error #15: Initializing libomp.dll` → Missing KMP_DUPLICATE_LIB_OK
- `Pointer argument cannot be accessed from Triton (cpu tensor?)` → Tensors not on CUDA

---

## 🔧 FLA LAYER API GOTCHAS

**These will bite you if you don't know them:**

```python
# 1. RWKV6Attention returns a TUPLE, not a tensor
out_rwkv, _, _ = self.rwkv6(norm_x)  # returns (output, attn_weights, past_kv)

# 2. Mamba2 returns just a tensor
out_mamba = self.mamba2(norm_x)  # returns tensor directly

# 3. Mamba2 num_heads must follow this formula:
#    num_heads = (expand * hidden_size) / head_dim
mamba_expand = 2
mamba_head_dim = 64
mamba_heads = (mamba_expand * hidden_size) // mamba_head_dim  # e.g., (2*128)/64 = 4

# 4. RWKV6 key_dim = hidden_size * 0.5 must be divisible by num_heads
#    For hidden=128, key_dim=64, num_heads must divide 64 (1,2,4,8,16,32,64)
```

---

## 🐧 LINUX PRODUCTION SETUP (For Larger Models)

**On Linux, install optional CUDA kernel packages for ~2-3x faster Mamba2:**

```bash
pip install causal-conv1d mamba-ssm
```

**No code changes required** - FLA auto-detects and uses faster kernels when available.

These packages don't compile on Windows (MSVC incompatibility), but the same model code
runs on both platforms. Windows uses Triton fallback; Linux can use native CUDA kernels.

---

## Core Requirement

**Two distinct components using FLA implementations:**
- **RWKV-6** from `fla/fla/layers/rwkv6.py` → `RWKV6Attention`
- **Mamba-2** from `fla/fla/layers/mamba2.py` → `Mamba2`

**Both must exist as separate, identifiable code paths with separate parameters.**

---

## ⚠️ THE ACTUAL ARCHITECTURE: Parallel Hybrid Block

**Every agent has gotten this wrong. This is the SIXTH time explaining it.**

### The Design: RWKV-6 and Mamba-2 Run IN PARALLEL Within Each Block

```
Input x
    │
    ├──────────────────┬──────────────────┐
    │                  │                  │
    ▼                  ▼                  │
  RWKV-6            Mamba-2              │ (Skip Connection)
    │                  │                  │
    ▼                  ▼                  │
  × rwkv_gain       × mamba_gain         │
    │                  │                  │
    └────────┬─────────┘                  │
             │                            │
             ▼                            │
           SUM ◄──────────────────────────┘
             │
             ▼
           FFN
             │
             ▼
         Output
```

### The Code (This Is What We're Building)

```python
import torch
import torch.nn as nn
from fla.layers.rwkv6 import RWKV6Attention
from fla.layers.mamba2 import Mamba2

class ParallelHybridBlock(nn.Module):
    """
    V4 Architecture: RWKV-6 and Mamba-2 running IN PARALLEL.
    No Grounding. No RWKV-7. Pure Hybrid.
    """
    def __init__(self, hidden_size, n_head=8):
        super().__init__()
        self.ln = nn.RMSNorm(hidden_size)
        
        # --- BRANCH 1: RWKV-6 ---
        self.rwkv6 = RWKV6Attention(hidden_size=hidden_size)
        
        # --- BRANCH 2: MAMBA-2 (SSD) ---
        self.mamba2 = Mamba2(hidden_size=hidden_size)
        
        # --- LEARNED FUSION ---
        # These allow the model to decide how much to trust each path
        self.rwkv_gain = nn.Parameter(torch.ones(hidden_size) * 0.5)
        self.mamba_gain = nn.Parameter(torch.ones(hidden_size) * 0.5)
        
        # Standard Feed-Forward
        self.ffn_ln = nn.RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=False)
        )

    def forward(self, x):
        # 1. Norm
        norm_x = self.ln(x)
        
        # 2. PARALLEL Computation - both kernels run on same input
        out_rwkv = self.rwkv6(norm_x)
        out_mamba = self.mamba2(norm_x)
        
        # 3. Parallel Fusion with skip connection
        x = x + (self.rwkv_gain * out_rwkv) + (self.mamba_gain * out_mamba)
        
        # 4. Standard FFN
        x = x + self.ffn(self.ffn_ln(x))
        
        return x


class HybridModel(nn.Module):
    """Stack of ParallelHybridBlocks"""
    
    def __init__(self, vocab_size=10000, hidden_size=128, n_layers=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # All layers are identical ParallelHybridBlocks
        self.layers = nn.ModuleList([
            ParallelHybridBlock(hidden_size) for _ in range(n_layers)
        ])
        
        self.norm_out = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Tied embeddings
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        return self.head(x)
```

### Why This Breaks the 7.0 Loss Wall

1. **Recurrence vs. Selectivity:** RWKV-6 provides smooth decay memory (persona over 2048+ tokens). Mamba-2 provides selective memory (snap to new instructions).

2. **Gradient Independence:** Running in parallel means A_log gradients (Mamba) and time_decay gradients (RWKV) don't compete.

3. **Correct Parameter Groups:** Logging works because 'rwkv' and 'mamba' are in separate named parameters.

### What NOT To Build

**WRONG (Sequential sandwich):**
```python
# DO NOT DO THIS
self.layers = [
    RWKV6Block(),      # Layer 0
    Mamba2Block(),     # Layer 1
    Mamba2Block(),     # ...
    RWKV6Block(),      # Layer N
]
```

**CORRECT (Parallel in every block):**
```python
# DO THIS
self.layers = [
    ParallelHybridBlock(),  # RWKV6 + Mamba2 together
    ParallelHybridBlock(),  # RWKV6 + Mamba2 together
    # ...
]
```

---

## Layer Size Reality Check

At 128 hidden dimension:
- **RWKV6 layer: ~700K params**
- **Mamba2 layer: ~67K params**

**This is a 10x difference!**

For equal parameter counts, we need **10x more Mamba2 layers than RWKV6 layers**.

---

## Exact Layer Configurations

### RWKV6 Layer (128 hidden) - ~700K params:

```python
RWKV6_Layer_128 = {
    'time_mixing': {
        'time_first': 128,      # 128 params
        'time_decay': 128,      # 128 params
        'receptance': 128×128,  # 16,384 params
        'key': 128×128,         # 16,384 params
        'value': 128×128,       # 16,384 params
        'gate': 128×128,        # 16,384 params
        'output': 128×128,      # 16,384 params
    },
    'channel_mixing': {
        'key': 128×(128×2),     # 32,768 params (2x expansion)
        'value': (128×2)×128,   # 32,768 params
        'receptance': 128×128,  # 16,384 params
        'gate': 128×128,        # 16,384 params
    },
    'layer_norms': 128×2,       # 256 params
    'Total': ~700,000 params
}
```

### Mamba2 Layer (128 hidden) - ~67K params:

```python
Mamba2_Layer_128 = {
    'in_proj': 128×256,         # 32,768 params (2x expansion)
    'out_proj': 256×128,        # 32,768 params
    'ssm': {
        'A': 8×8,               # 64 params (state 8×8)
        'B': 128×8,             # 1,024 params
        'C': 8×128,             # 1,024 params
        'D': 128,               # 128 params
    },
    'conv1d': 128×4,            # 512 params (kernel=4)
    'layer_norm': 128,          # 128 params
    'Total': ~67,000 params
}
```

---

## WHAT WE ACTUALLY BUILT: Parallel Block Architecture (Implemented 2026-01-09)

**This is what exists in code:**

```
HybridModel (GF-MH - Winner)
├─ Embedding: vocab → 128 hidden
├─ Block 0: [RWKV-6] ∥ [Mamba-2] → GF (gate) → FFN
├─ Block 1: [RWKV-6] ∥ [Mamba-2] → GF (gate) → FFN
├─ Block 2: [RWKV-6] ∥ [Mamba-2] → GF (gate) → FFN
├─ Block 3: [RWKV-6] ∥ [Mamba-2] → GF (gate) → FFN
├─ Block 4: [RWKV-6] ∥ [Mamba-2] → GF (gate) → FFN
├─ Block 5: [RWKV-6] ∥ [Mamba-2] → GF (gate) → FFN
├─ Block 6: [RWKV-6] ∥ [Mamba-2] → GF (gate) → FFN
├─ Block 7: [RWKV-6] ∥ [Mamba-2] → GF (gate) → FFN
├─ LayerNorm
└─ Output Head

Key Facts:
- 8 blocks total (not "2 RWKV + 10 Mamba")
- Each block has BOTH 1 RWKV-6 and 1 Mamba-2 running in parallel
- Gate-based fusion (GF) learns optimal weighting per position
- GF-MH uses gate_init=0.3 (70% Mamba at start)
- Total: ~3.5M parameters

File: hybrid_v4_ratio.py (GF-MH is the winner)
```

### Visual: Single ParallelHybridBlock

```
┌──────────────────────────────────────────────────────┐
│            ParallelHybridBlock (i)                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Input: [batch, seq_len, 128]                       │
│         │                                            │
│         ▼                                            │
│      RMSNorm                                         │
│      ╱      ╲                                        │
│     ╱        ╲                                       │
│    ▼          ▼                                      │
│  RWKV-6    Mamba-2                 SKIP ────────┐   │
│    │          │                                  │   │
│    ├─ norm ───┤                                  │   │
│    ▼          ▼                                  │   │
│    │    gate  │    ◄── Gated Fusion             │   │
│    ├─────────┬┘     (learns blend at each       │   │
│    ▼         ▼       position)                   │   │
│  gate * rwkv + (1-gate) * mamba ◄────────┘      │   │
│         │                                        │   │
│         ▼                                        │   │
│      + SKIP connection                           │   │
│         │                                        │   │
│         ▼                                        │   │
│      RMSNorm                                     │   │
│         │                                        │   │
│        FFN (Linear→GELU→Linear)                  │   │
│         │                                        │   │
│         ▼                                        │   │
│       + SKIP (residual)                          │   │
│         │                                        │   │
│         ▼                                        │   │
│  Output: [batch, seq_len, 128]                   │   │
│                                                  │   │
└──────────────────────────────────────────────────────┘
```

### Visual: Full Model Data Flow

```
Input tokens [batch_size, seq_len]
    │
    ▼
Embedding → [batch_size, seq_len, 128]
    │
    ├──► Block 0: RWKV∥Mamba + GF fusion + FFN → [batch, seq, 128]
    │
    ├──► Block 1: RWKV∥Mamba + GF fusion + FFN → [batch, seq, 128]
    │
    ├──► Block 2: RWKV∥Mamba + GF fusion + FFN → [batch, seq, 128]
    │
    ├──► Block 3: RWKV∥Mamba + GF fusion + FFN → [batch, seq, 128]
    │
    ├──► Block 4: RWKV∥Mamba + GF fusion + FFN → [batch, seq, 128]
    │
    ├──► Block 5: RWKV∥Mamba + GF fusion + FFN → [batch, seq, 128]
    │
    ├──► Block 6: RWKV∥Mamba + GF fusion + FFN → [batch, seq, 128]
    │
    ├──► Block 7: RWKV∥Mamba + GF fusion + FFN → [batch, seq, 128]
    │
    ▼
LayerNorm → [batch, seq, 128]
    │
    ▼
Linear Head → [batch, seq, vocab_size=97]
    │
    ▼
Logits (softmax for inference)
```

### Contrast: Sequential vs Parallel

```
PROPOSED (Sequential - never built):
Input
  ↓
[RWKV-6 Layer 0]
  ↓
[Mamba-2 Layer 0]
  ↓
[Mamba-2 Layer 1]
  ↓
...
[Mamba-2 Layer 20]
  ↓
[RWKV-6 Layer 1]
  ↓
Output

ACTUAL (Parallel blocks - implemented):
Input
  ↓
[RWKV-6 + Mamba-2 in parallel] ← Block 0
  ↓
[RWKV-6 + Mamba-2 in parallel] ← Block 1
  ↓
[RWKV-6 + Mamba-2 in parallel] ← Block 2
  ↓
...
[RWKV-6 + Mamba-2 in parallel] ← Block 7
  ↓
Output

Key Difference:
- Sequential: Data flows through different layer types
- Parallel: Both layer types process same input, outputs are fused
```

---

## 5-8M Hybrid Prototypes (PROPOSED - Not Yet Implemented)

**These are theoretical sequential architectures for future exploration:**

### PROPOSED Prototype 1: Sequential Balanced 5M

```
SEQUENTIAL STACK (not parallel blocks):
RWKV6 → [Mamba2 ×21] → RWKV6
- 2 RWKV6 layers (1.4M)
- 21 Mamba2 layers (1.4M)

Embeddings: 1.28M (10K×128)
Output: 1.28M (tied)
Fusion: 33K
Total: ~4.4M (with tied)
Layer Ratio: 1:10.5 (RWKV6:Mamba2)
```

### PROPOSED Prototype 2: Sequential RWKV-Heavy 5M

```
SEQUENTIAL STACK (not parallel blocks):
RWKV6 → [Mamba2 ×5] → RWKV6 → [Mamba2 ×5] → RWKV6 × 2
```

---

## Training Configuration for 6GB VRAM

### Memory Budget for 5M Model:

```
Model weights (FP16): 5M × 2 bytes = 10MB
Optimizer states (Adam): 5M × 8 bytes = 40MB
Gradients (FP16): 5M × 2 bytes = 10MB
Activations (batch=32, seq=1024): ~200MB
Total: ~260MB → Plenty of room!

You can use batch size up to 256 or sequence length up to 8192!
```

---

## Test Suite: Hybrid Models Only

**Note:** We skip pure RWKV6 and pure Mamba2 training. Public metrics exist. We're here to build hybrids.

### ⚠️ PROPOSED Sequential Variants (Not Yet Implemented)

These show the original design concept. We tested **parallel blocks instead** (see above).

#### 1. PROPOSED: Sequential Balanced (5M)

```
SEQUENTIAL architecture (not parallel blocks):
RWKV6 → [Mamba2 ×21] → RWKV6
- 2 RWKV6 layers (1.4M)
- 21 Mamba2 layers (1.4M)
- Rest: Embed/Output/Fusion (2.2M)
- Total: ~5.4M
```

#### 2. PROPOSED: Sequential RWKV-Heavy (5M)

```
SEQUENTIAL architecture:
RWKV6 → [Mamba2 ×5] → RWKV6 → [Mamba2 ×5] → RWKV6 × 2
- 4 RWKV6 layers (2.8M)
- 10 Mamba2 layers (0.67M)
- Rest: Embed/Output/Fusion (1.53M)
- Total: ~5.0M
```

#### 3. PROPOSED: Sequential Mamba-Heavy (5M)

```
SEQUENTIAL architecture:
RWKV6 → [Mamba2 ×30]
- 1 RWKV6 layer (0.7M)
- 30 Mamba2 layers (2.0M)
- Rest: Embed/Output/Fusion (2.3M)
- Total: ~5.0M
```

**Status:** These are documented for future reference. Phase 2 chose **parallel block architecture** instead and benchmarked fusion+ratio variants. See hybrid_v4_ratio.py for actual implementation.

---

## Training Configuration

### Hyperparameters (Standard for All Experiments)

```python
training_config = {
    # Optimizer
    'optimizer': 'AdamW',
    'lr': 3e-4,
    'weight_decay': 0.1,
    'betas': (0.9, 0.95),
    
    # Schedule
    'warmup_steps': 2000,      # 2-4x longer for hybrids (V3 Cross-Ref 1.6)
    'lr_decay': 'cosine',
    'min_lr': 3e-5,            # 10% of peak
    
    # Batch
    'batch_size': 32,
    'grad_accum_steps': 4,     # effective batch = 128
    'max_seq_len': 1024,
    
    # Training
    'max_steps': 100_000,
    'eval_every': 100,         # Val loss every 100 steps
    'log_every': 10,           # Train loss every 10 steps
    'save_every': 10_000,
    
    # Regularization
    'dropout': 0.0,            # none for small models
    'grad_clip': 1.0,
}
```

### Parameter Groups (Differential LR)

**From V3 Section 2.15, 9.6:** RWKV and Mamba components may need different learning rates.

```python
def get_parameter_groups(model, base_lr=3e-4):
    """
    Separate RWKV and Mamba parameters for differential LR.
    Mamba may need 1.5-3x higher LR initially.
    """
    rwkv_params = []
    mamba_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'rwkv' in name.lower():
            rwkv_params.append(param)
        elif 'mamba' in name.lower():
            mamba_params.append(param)
        else:
            other_params.append(param)
    
    return [
        {'params': rwkv_params, 'lr': base_lr, 'name': 'rwkv'},
        {'params': mamba_params, 'lr': base_lr * 2.0, 'name': 'mamba'},  # 2x for Mamba
        {'params': other_params, 'lr': base_lr, 'name': 'other'},
    ]

# Usage:
# optimizer = AdamW(get_parameter_groups(model, lr=3e-4))
```

### Plateau Response Protocol

**From V3 Cross-Ref 1.3, 1.8:**

| Plateau Duration | Action |
|------------------|--------|
| 1-10% of training | Continue - normal |
| 10-20% of training | Reduce LR by 30-50%, train 10-20% more |
| >20% with no val improvement | Likely converged, stop |

**Stop immediately if:** 2+ LR reductions with no improvement.

### Logging Requirements

**MUST log both train AND val loss:**
```python
# Every 10 steps:
log(f"Step {step} | Train Loss: {train_loss:.4f}")

# Every 100 steps:
log(f"Step {step} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | RWKV/Mamba ratio: {ratio:.2f}")
```

### Tokenizer

```python
tokenizer_config = {
    'type': 'BPE',  # or sentencepiece, or CharTokenizer for quick tests
    'vocab_size': 10_000,
    'source': 'trained on test dataset or use existing tokenizer_v030.py',
}
```

**Important:** Same config for ALL hybrid experiments. No tuning per-model.

---

## Fusion Mechanisms

### Option 1: Concatenate + Project (Cheapest)

```python
# RWKV_out: [batch, seq, 128]
# Mamba_out: [batch, seq, 128]
combined = torch.cat([rwkv_out, mamba_out], dim=-1)  # 256 dim
fused = nn.Linear(256, 128)(combined)  # 33K params
```

### Option 2: Weighted Sum (Learnable)

```python
alpha = nn.Parameter(torch.tensor(0.5))
fused = alpha * rwkv_out + (1 - alpha) * mamba_out  # 1 param!
```

### Option 3: Gated Fusion (Light)

```python
gate = torch.sigmoid(nn.Linear(256, 1)(combined))  # 257 params
fused = gate * rwkv_out + (1 - gate) * mamba_out
```

### Option 4: Residual Fusion

```python
base = rwkv_out  # Use RWKV as base
residual = mamba_out  # Mamba as correction
fused = base + nn.Linear(128, 128)(residual)  # 16K params
```

---

## [CONTINUED - MORE COMING]

---

## What NOT To Do

- Do NOT substitute RWKV-7 for RWKV-6
- Do NOT skip Mamba and call something else "selective"
- Do NOT build all 5 prototypes at once
- Do NOT jump to larger scales before validating at 5-8M
- Do NOT assume - ASK if unclear

---

*This document is the spec. Build what it says.*
