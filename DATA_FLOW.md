# GroundThink V4 Data Flow & Architecture Diagram

**Purpose:** Visualize how data moves through the model and understand each component's role.

---

## Level 1: Single Token → Output (Conceptual)

```
Input Token (e.g., "a")
    │
    ▼
Embedding (token → 128-dim vector)
    │
    ├─────────────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
┌──────────────────────────────────────────────┐      │
│          ParallelHybridBlock 0               │      │ (Residual)
├──────────────────────────────────────────────┤      │
│ Input: [batch, seq_len, 128]                 │      │
│   │                                          │      │
│   ├──→ LayerNorm                             │      │
│   │      │                                   │      │
│   │   ┌──┴──┐                                │      │
│   │   │     │                                │      │
│   │   ▼     ▼                                │      │
│   │ RWKV  Mamba (run in parallel)           │      │
│   │   │     │                                │      │
│   │   └──┬──┘                                │      │
│   │      │                                   │      │
│   │      ▼                                   │      │
│   │  Gated Fusion (learned blend)           │      │
│   │      │                                   │      │
│   ├──────┴───────────────────────────────┐  │      │
│   │                                      │  │      │
│   └───────────────────┬──────────────────┘  │      │
│                       │ (add residual)      │      │
│                       └──────────┬───────────┘      │
│                                  │                  │
│                    ┌─────────────┘                  │
│                    │                                │
│                    ▼                                │
│             LayerNorm + FFN (2x expansion)        │
│                    │                                │
│                    └──────────┬──────────────────┐  │
│                               │                  │  │
│                               ▼                  │  │
│                          (add residual)          │  │
│                               │                  │  │
│                               └──────────────────┘  │
│                                                     │
│  Output: [batch, seq_len, 128]                     │
└──────────────────────────────────────────────────────┘
    │
    ├─────→ (repeat for Blocks 1-7)
    │
    ▼
LayerNorm
    │
    ▼
Linear Head (128 → vocab_size)
    │
    ▼
Logits (softmax for inference)
```

---

## Level 2: Full Model Architecture

```
Tokens: [batch_size=64, seq_len=64]
    │
    ▼
Embedding Layer (vocab_size=97 → 128 hidden)
    │
    ▼
┌─────────────────────────────────────────┐
│  Block 0: RWKV∥Mamba + Gated Fusion     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Block 1: RWKV∥Mamba + Gated Fusion     │
└─────────────────────────────────────────┘
    │
    ▼
        ... (Blocks 2-6)
    │
    ▼
┌─────────────────────────────────────────┐
│  Block 7: RWKV∥Mamba + Gated Fusion     │
└─────────────────────────────────────────┘
    │
    ▼
Final LayerNorm
    │
    ▼
Output Head (128 → 97 vocab)
    │
    ▼
Logits: [batch_size, seq_len, vocab_size]
    │
    ▼
Loss (cross-entropy with targets)
    │
    ▼
Backprop → Optimizer Update
```

**Key:** All 8 blocks are identical (same weights initialized differently, but same architecture).

---

## Level 3: Inside One ParallelHybridBlock (Detailed)

```
Input x: [batch, seq_len, 128]
    │
    ▼
RMSNorm (normalize activations)
    │
    ├─────────────────────────┬──────────────────────┐
    │                         │                      │
    │                         │ (Store for skip)     │
    │                         │                      │
    ▼                         │                      │
RWKV-6 Layer                  │                      │
    │                         │                      │
    │ (returns tuple)          │                      │
    │ out_rwkv: [batch, seq, 128]
    │                         │                      │
    └─────────┬───────────────┘                      │
              │                                      │
    ┌─────────┘                                      │
    │                                                │
    ▼                                                │
Mamba-2 Layer                  ▲                     │
    │                          │                     │
    │ out_mamba: [batch, seq, 128]
    │                          │                     │
    └─────────┬────────────────┼─────────────────────┘
              │                │
              ▼                │
      ┌──────────────┐        │
      │ Gate Network │        │
      │ out: sigmoid │        │
      └──────┬───────┘        │
             │                │
             ▼                ▼
     gate * rwkv_out + (1-gate) * mamba_out
             │
             ▼
         Fusion Output: [batch, seq, 128]
             │
             ├─ Add Skip Connection (residual from RMSNorm input)
             │
             ▼
         x = x + fused_out
             │
             ▼
         RMSNorm (pre-FFN)
             │
             ▼
    Linear(128 → 512) + GELU activation
             │
             ▼
    Linear(512 → 128)
             │
             ├─ Add Skip Connection (residual from pre-FFN)
             │
             ▼
    Output: [batch, seq_len, 128]
```

---

## Level 4: RWKV-6 Component (Internal)

```
Input: [batch, seq_len, 128]
    │
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼                                             │
Time Mixing (Recurrent Memory)                    │
    │                                             │
    ├─→ Receptance: Linear(128 → 128)            │
    ├─→ Key: Linear(128 → 128)                   │
    ├─→ Value: Linear(128 → 128)                 │
    │                                             │
    └─→ Combine with exponential decay:          │
        new_state = (value * key) * decay + old_state
        output = receptance * state              │
             │                                    │
             └────────────┐                       │
                          │                       │
    ┌─────────────────────┘                       │
    │                                             │
    ▼                                             │
Channel Mixing (FFN-like)                         │
    │                                             │
    ├─→ Key: Linear(128 → 256)                   │
    ├─→ Value: Linear(256 → 128)                 │
    │                                             │
    └─→ output = value * gate(key)                │
             │                                    │
             └───────────────┬────────────────────┘
                             │
                             ▼
                    Output: [batch, seq, 128]

**Key Insight:** RWKV maintains running state with exponential decay.
No attention matrix; purely recurrent.
```

---

## Level 5: Mamba-2 Component (Internal)

```
Input: [batch, seq_len, 128]
    │
    ▼
Input Projection (Linear): 128 → 256
    │ (expand internal state)
    │
    ▼
Selective State Space
    │
    ├─→ Project input to SSM input
    │
    ├─→ State Evolution (content-dependent):
    │   new_state = A * state + B * input
    │   where A, B depend on input content
    │
    ├─→ Output: C_state (C projection)
    │
    ├─→ Skip Connection (residual within Mamba)
    │
    └─→ Output: [batch, seq_len, 256]
    │
    ▼
Output Projection (Linear): 256 → 128
    │ (compress back)
    │
    ▼
Output: [batch, seq_len, 128]

**Key Insight:** Mamba's state transition (A, B, C matrices) adapts
based on input. This is "selective" — different tokens
remember different things.
```

---

## Level 6: Gated Fusion Mechanism

```
RWKV Output: [batch, seq_len, 128]
Mamba Output: [batch, seq_len, 128]
    │
    ├────────────────┬────────────────┐
    │                │                │
    ▼                │                │
Concatenate         │                │
    │                │                │
    ▼                │                │
Linear(256 → 1)    │                │
+ Sigmoid            │                │
    │                │                │
    ▼                │                │
Gate: [batch, seq, 1]
    │                │                │
    ├────────────────┘                │
    │                                 │
    ▼                                 │
gate * rwkv_out + (1-gate) * mamba_out
    │                                 │
    ├─────────────────────────────────┘
    │
    ▼
Fused Output: [batch, seq_len, 128]

**Per-Position Blending:** Different tokens choose different blend ratios.
Gate initialized to 0.3 (70% Mamba, 30% RWKV) for GF-MH.
```

---

## Data Shapes Through the Model

```
Stage                          Shape              Params
─────────────────────────────────────────────────────────
Input Tokens                   [64, 64]           —
↓
Embedding                      [64, 64, 128]      1.28M
↓
Block 0:
  ├─ RWKV-6                    [64, 64, 128]      ~700K
  ├─ Mamba-2                   [64, 64, 128]      ~67K
  ├─ Fusion Gate               [64, 64, 1]        ~256
  └─ Output                    [64, 64, 128]      
↓
Blocks 1-7 (same as Block 0)   [64, 64, 128]      8 × (~700K + ~67K)
↓
Output Projection              [64, 64, 97]       1.28M (tied)
↓
Loss (scalar)                  []                 —

Total: ~3.5M parameters
```

---

## Gradient Flow (Training)

```
Loss (scalar)
    │
    ▼
dL/d(logits)
    │
    ▼
Backprop through Output Head
    │
    ▼
dL/d(Block_7 output)
    │
    ├─────────────────┬──────────────────┐
    │                 │                  │
    ▼                 ▼                  ▼
dL/dRWKV        dL/dMamba         dL/dGate
    │                 │                  │
    └────────┬────────┴──────────────────┘
             │
             ▼
     (repeat for Blocks 6-0)
             │
             ▼
     dL/d(Embedding)
             │
             ▼
     Optimizer Update

**Key:** RWKV and Mamba gradients flow independently → no competition.
```

---

## Memory Usage (Batch=64, Seq=64)

```
Component                          Size
─────────────────────────────────────────
Model Weights (FP32)               ~14 MB
Gradients (FP32)                   ~14 MB
Optimizer States (Adam: m, v)      ~56 MB
─────────────────────────────────────────
Activations (forward pass)         ~50 MB
  ├─ Embeddings                    ~2 MB
  ├─ Block outputs (×8)            ~40 MB
  └─ Intermediate (norms, FFN)     ~8 MB
─────────────────────────────────────────
TOTAL                              ~184 MB

Headroom on 6GB VRAM: ~5.8GB free
Can scale: batch→128, seq→256 (still fits)
```

---

## Inference Flow (No Training)

```
Input: "The quick brown"
    │
    ▼
Tokenize → [1, 88, 45, 921]
    │
    ▼
Embed → [4, 1, 128]
    │
    ▼
Block 0 forward pass (accumulate state)
    │
    ├─ RWKV state updated (exponential decay)
    ├─ Mamba state updated (content-dependent)
    └─ Output fused
    │
    ▼
Blocks 1-7 (same, updating internal states)
    │
    ▼
Logits for next token: [1, 97]
    │
    ▼
Sample from distribution → "fox"
    │
    ▼
Add to sequence, repeat

**Key:** States persist across tokens → context-aware generation
```

---

## Comparison: Data Flow Between Variants

### GF (Balanced Gated Fusion)
```
Gate initialized to 0.5
→ Equal RWKV/Mamba blend at start
→ Gate learns to adjust
→ Final: approximately 50-50 by end of training
```

### GF-MH (Mamba-Heavy, Phase 2 Winner)
```
Gate initialized to 0.3
→ Favors Mamba at start (70% Mamba, 30% RWKV)
→ Gate can shift, but starts biased toward Mamba
→ Final: Mamba benefits from head start
→ Result: Better val loss (-1.8% vs balanced)
```

### GF-RH (RWKV-Heavy)
```
Gate initialized to 0.7
→ Favors RWKV at start (70% RWKV, 30% Mamba)
→ Result: Worse than balanced (-0.3%)
→ Interpretation: RWKV's 10× param cost not worth the extra computation
```

---

## Why This Architecture Matters

### Linear Complexity
```
Standard Transformer:
  Attention: O(n²) — matrix of all pairs
  For 64 tokens: 4,096 operations
  For 16K tokens: 256 million operations

RWKV + Mamba:
  Both: O(n) — single pass over sequence
  For 64 tokens: 64 operations
  For 16K tokens: 16K operations
```

### Parallel Gradient Flow
```
If sequential (Transformer-style):
  Gradients back through 8 RWKV + 8 Mamba layers

Parallel (GroundThink):
  RWKV gradients: independent path
  Mamba gradients: independent path
  → No gradient competition
  → Easier to balance components
```

### Selective vs Smooth Memory
```
RWKV: "Remember everything, fade exponentially"
  Good for: Long narratives, temporal patterns
  Bad for: Sudden context shifts

Mamba: "Remember what matters, forget the rest"
  Good for: Context switching, irrelevant info filtering
  Bad for: Maintaining long-term subtle patterns

Hybrid: "Both strategies, blend dynamically"
  → Get benefits of each
```

---

## References

- **RWKV Details:** [V4_DESIGN.md - Layer Size Reality Check](V4_DESIGN.md)
- **Mamba Details:** [ONBOARDING.md - Part 1, Section "Mamba-2"](ONBOARDING.md)
- **Fusion Strategy:** [V4_DESIGN.md - Fusion Mechanisms](V4_DESIGN.md)
- **Implementation Code:** [hybrid_v4_ratio.py](hybrid_v4_ratio.py)
- **Phase 2 Results:** [CHANGELOG.md - Section 4.2-Alpha](CHANGELOG.md)

---

**Last Updated:** 2026-01-10  
**For Questions:** See [ONBOARDING.md Part 10](ONBOARDING.md#part-10-the-librarians-note)
