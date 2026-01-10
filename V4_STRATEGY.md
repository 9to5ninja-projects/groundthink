# V4 Strategy Document - Task Backlog

**Created:** 2026-01-08  
**Updated:** 2026-01-09  
**Repository:** https://github.com/9to5ninja-projects/groundthink  
**Purpose:** Ordered queue of tasks to be completed one at a time  
**Current Goal:** Build RWKV-6 + Mamba-2 hybrid at 5M scale, find best configuration

---

## Key Insight

At 5-8M parameters, Mamba2 layers are **~10x more parameter-efficient** than RWKV6 layers.

- RWKV6 layer (128 hidden): ~700K params
- Mamba2 layer (128 hidden): ~67K params

This dramatically changes hybrid ratio calculations.

**Why we skip pure model training:**
- Public metrics exist for both RWKV6 and Mamba2 at various scales
- Training them here would use different methods than our hybrids (unfair comparison)
- We already know 5M pure models are weak - that's not the research question
- **The goal is to build hybrids, not prove pure models suck**

---

## Validation Gates (From V3 Section 9.5)

**Every gate must pass before proceeding. No exceptions.**

| Gate | Test | Pass | Warn | Fail |
|------|------|------|------|------|
| G1 | Forward pass | No NaN, correct shapes | - | NaN or shape mismatch |
| G2 | Init entropy | 2.0-5.0 at step 0 | 6.0-7.0 | <1.0 or >8.0 |
| G3 | Train 1k steps | Loss decreasing, grad 0.5-1.5 | Grad 1.5-3.0 | Grad >5.0 or loss increasing |
| G3.5 | State health | Cosine <0.99, SVD >0.5, sat <30% | Cosine 0.95-0.99 | Cosine >0.99 (frozen) |
| G4 | Component balance | Gradient ratio 0.3-3.0 | 0.1-0.3 or 3-10 | <0.1 or >10 (imbalance) |

---

## Stopping Criteria (From V3 Section 9.7, Cross-Ref 1.2)

### Stop Immediately If:
- **Val loss increasing >5-10%** while train decreases (overfitting)
- **2+ LR drops** with no improvement
- **Oscillating loss** (up-down >0.5) - architecture conflict
- **One component's activations collapse** to constant
- **Gradient ratio <0.1 or >10** - one component is dead

### Continue If:
- Val loss shows "heartbeat" (small dips every few hundred steps)
- Both components have gradient variance
- Training loss still has tiny downward slope on log scale

---

## Two Types of Loss (BOTH Required)

1. **Train Loss** - Log every step
2. **Val Loss** - Log every 100-1000 steps

Both must be tracked, plotted, and checked for divergence.

---

## V3 Materials (Archived)

**All V3 documentation and code have been moved to `archive/` for reference.**

V3 was scrapped because agents built RWKV-7 instead of RWKV-6. V4 uses FLA library implementations (RWKV6Attention + Mamba2) exclusively.

**V3 files in archive:**
- Documentation: V3_STRATEGY.md, V3_BUILD_LOG.md, V3_CROSS_REFERENCE.md, V3_RESEARCH_NOTES.md, V3_DEPRECATED.md
- Code: train_v030.py, data_v030.py, layers_v030.py, tokenizer_v030.py, layers_v020.py
- Old tests and diagnostics: check_*.py, test_*.py, verify_*.py, trace_*.py, gate_g35_diagnostic.py
- Build scripts: build_causal.sh, build_step5.sh, fla_replacements.py, rwkv6_layer.py
- V2 materials: V2_INSTRUCTIONS.md, design_v2_notes.txt, original_design_notes.txt

**These can be referenced but should not be copied forward without careful review.**

---

## How This Document Works

**This is a BACKLOG, not a to-do list for one agent.**

1. Agent checks V4_HANDOFF.md for active task
2. If no active task: pick NEXT task from this backlog (in order)
3. Copy task to V4_HANDOFF.md "Active Task" section
4. **IMMEDIATELY use `manage_todo_list` tool** to create sub-task checklist
5. Work on that ONE task until complete
6. When done: clear from handoff, mark complete here, prepare for next agent

### Why the Todo Tool is Required

The `manage_todo_list` tool:
- Keeps agent focused on ONE task
- Provides user visibility into progress
- Prevents context drift and scope creep
- Creates natural checkpoints for multi-session work

**First action after reading handoff = write todo list. No exceptions.**

**Do NOT:**
- Work on multiple tasks at once
- Skip ahead in the backlog

### When Stuck or Uncertain

**ASK THE USER. Do not guess.**

---

## Task Backlog

### Phase 1: Foundation (Must Complete First)

| # | Task | Status | Depends On | Time |
|---|------|--------|------------|------|
| 1 | Verify data_v030.py + tokenizer_v030.py work | ✅ COMPLETE | - | 30 min |
| 2 | Create/Validate Test Dataset (train+val split) | ✅ COMPLETE | Task 1 | 1 hr |
| 3 | Build First Hybrid (ParallelHybridBlock) | ✅ COMPLETE | - | 2 hr |
| 4 | Define Training Configuration | ⬜ PENDING | Task 3 | 30 min |
| 5 | Pass Gate G1-G2 (forward pass, init entropy) | ⬜ PENDING | Task 3 | 30 min |
| 6 | Add Component Gradient Logging | ⬜ PENDING | Task 5 | 30 min |
| 7 | Train 1K steps, Pass Gate G3-G4 | ⬜ PENDING | Tasks 2, 4, 6 | 30 min |
| 8 | Run State Health Diagnostic (Gate G3.5) | ⬜ PENDING | Task 7 | 30 min |
| 9 | Train First Hybrid (100K steps) | ⬜ PENDING | Task 8 | 2-4 hrs |

**Purpose:** Get a working hybrid trained WITH proper validation at each step.

**Gate:** Phase 1 complete when:
- All gates G1-G4, G3.5 pass
- 100K steps complete without stopping criteria triggered
- Both train AND val loss curves show healthy descent

### Phase 2: Hybrid Ratio Comparison (After Phase 1)

| # | Task | Status | Depends On | Time |
|---|------|--------|------------|------|
| 10 | Build Hybrid RWKV-Heavy (4 RWKV6 + 10 Mamba2) | ⬜ PENDING | Task 9 | 1 hr |
| 11 | Pass Gates G1-G3.5 for RWKV-Heavy | ⬜ PENDING | Task 10 | 30 min |
| 12 | Build Hybrid Mamba2-Heavy (1 RWKV6 + 30 Mamba2) | ⬜ PENDING | Task 9 | 1 hr |
| 13 | Pass Gates G1-G3.5 for Mamba2-Heavy | ⬜ PENDING | Task 12 | 30 min |
| 14 | Train Both Variants (100K steps each) | ⬜ PENDING | Tasks 11, 13 | 4-8 hrs |
| 15 | Analyze Ratio Results (compare val loss) | ⬜ PENDING | Task 14 | 1 hr |

**Gate:** Phase 2 complete when best hybrid ratio identified by lowest final val loss.

### Phase 3: Fusion Mechanisms (After Phase 2)

| # | Task | Status | Depends On | Time |
|---|------|--------|------------|------|
| 16 | Implement 4 Fusion Variants | ⬜ PENDING | Task 15 | 2 hrs |
| 17 | Train Best Ratio with Each Fusion | ⬜ PENDING | Task 16 | 4-8 hrs |
| 18 | Analyze Fusion Results (compare val loss) | ⬜ PENDING | Task 17 | 1 hr |

**Gate:** Phase 3 complete when best fusion mechanism identified by lowest final val loss.

### Phase 4: Scaling Test (After Phase 3)

| # | Task | Status | Depends On | Time |
|---|------|--------|------------|------|
| 19 | Scale Best Config to 8M | ⬜ PENDING | Task 18 | 2 hrs |
| 20 | Pass Gates G1-G3.5 for 8M | ⬜ PENDING | Task 19 | 1 hr |
| 21 | Train 8M Model (100K steps) | ⬜ PENDING | Task 20 | 4-8 hrs |
| 22 | Compare 5M vs 8M Scaling | ⬜ PENDING | Task 21 | 1 hr |
| 23 | Document Final Findings | ⬜ PENDING | Task 22 | 1 hr |

**Legend:** ⬜ PENDING | 🔄 IN HANDOFF | ✅ COMPLETE

---

## Phase 1 Task Details

### Task 1: Verify data_v030.py + tokenizer_v030.py Work

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Check if V3 data infrastructure is reusable

**Requirements:**
- Load `data_v030.py` and confirm `StatefulDataset` instantiates
- Load `tokenizer_v030.py` and confirm `CharTokenizer` or `BPETokenizer` works
- Verify train/val split logic is intact

**Acceptance Criteria:**
- [ ] `StatefulDataset` loads test data without error
- [ ] Tokenizer encodes/decodes correctly
- [ ] Document any needed fixes in V4_BUILD_LOG.md

---

### Task 2: Create/Validate Test Dataset

**Status:** ⬜ PENDING  
**Time:** ~1 hour  
**Scope:** Prepare consistent data for all experiments

**Requirements:**
- 1M tokens minimum
- Consistent train/val split (90/10) using StatefulDataset
- Document source and preprocessing
- **MUST track both train AND val loss**

**Acceptance Criteria:**
- [ ] Dataset file exists and loads correctly
- [ ] Train/val split verified (no overlap)
- [ ] Tokenizer vocab size documented
- [ ] Document in V4_BUILD_LOG.md

---

### Task 3: Build First Hybrid (Parallel Block Architecture)

**Status:** ✅ COMPLETE (2026-01-09)  
**Output:** `hybrid_v4.py`  
**Time:** ~2 hours  
**Scope:** Create working RWKV6 + Mamba2 PARALLEL hybrid model

**⚠️ READ V4_DESIGN.md "THE ACTUAL ARCHITECTURE" SECTION**

**Architecture: Parallel Hybrid Blocks**
- Each block contains BOTH RWKV6 AND Mamba2 running in parallel
- Learned fusion gains (rwkv_gain, mamba_gain)
- 8 blocks × ~650K params/block = ~5.2M + embeddings

**Implementation (CORRECT - Parallel in Every Block):**
```python
import torch
import torch.nn as nn
from fla.layers.rwkv6 import RWKV6Attention
from fla.layers.mamba2 import Mamba2

class ParallelHybridBlock(nn.Module):
    """RWKV-6 and Mamba-2 running IN PARALLEL within each block."""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.ln = nn.RMSNorm(hidden_size)
        
        # BRANCH 1: RWKV-6
        self.rwkv6 = RWKV6Attention(hidden_size=hidden_size)
        
        # BRANCH 2: MAMBA-2
        self.mamba2 = Mamba2(hidden_size=hidden_size)
        
        # LEARNED FUSION
        self.rwkv_gain = nn.Parameter(torch.ones(hidden_size) * 0.5)
        self.mamba_gain = nn.Parameter(torch.ones(hidden_size) * 0.5)
        
        # FFN
        self.ffn_ln = nn.RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=False)
        )

    def forward(self, x):
        norm_x = self.ln(x)
        
        # PARALLEL computation
        out_rwkv = self.rwkv6(norm_x)
        out_mamba = self.mamba2(norm_x)
        
        # Fusion with skip connection
        x = x + (self.rwkv_gain * out_rwkv) + (self.mamba_gain * out_mamba)
        
        # FFN
        x = x + self.ffn(self.ffn_ln(x))
        return x


class HybridModel(nn.Module):
    def __init__(self, vocab_size=10000, hidden_size=128, n_layers=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            ParallelHybridBlock(hidden_size) for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.embed.weight
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        return self.head(x)
```

**Acceptance Criteria:**
- [ ] Model instantiates without error
- [ ] Forward pass works on sample input
- [ ] Parameter count verified (~5M)
- [ ] Document in V4_BUILD_LOG.md

---

### Task 4: Define Training Configuration

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Lock down hyperparameters AFTER model is built (need VRAM measurements)

**Requirements:**
- Learning rate schedule (warmup + decay)
- Optimizer (AdamW recommended)
- Batch size (fit in 6GB VRAM) - **must test with actual model**
- Gradient accumulation steps
- **Val loss eval frequency** (every 100-1000 steps)
- Checkpoint frequency

**Recommended Starting Point:**
```python
config = {
    'lr': 3e-4,
    'warmup_steps': 1000,
    'optimizer': 'AdamW',
    'weight_decay': 0.1,
    'batch_size': 32,        # ADJUST based on Task 3 VRAM test
    'grad_accum': 4,
    'max_steps': 100_000,
    'eval_every': 100,       # Val loss every 100 steps
    'save_every': 10_000,
    'log_every': 10,         # Train loss every 10 steps
}
```

**Acceptance Criteria:**
- [ ] Config documented in V4_DESIGN.md
- [ ] Same config used for ALL hybrid experiments
- [ ] VRAM usage verified < 6GB with actual model

---

### Task 5: Pass Gate G1-G2

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Validate model before adding gradient logging

**G1 - Forward Pass:**
- [ ] No NaN in outputs
- [ ] Correct output shapes
- [ ] VRAM fits in 6GB

**G2 - Init Entropy:**
- [ ] State entropy at step 0 between 2.0-5.0
- [ ] Warn if 6.0-7.0
- [ ] FAIL if <1.0 or >8.0

---

### Task 6: Add Component Gradient Logging

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Must verify BOTH RWKV6 and Mamba2 receive gradients

**From V3 Section 9.4:**
```python
def log_component_gradients(model):
    rwkv_grad_norms = []
    mamba_grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'rwkv' in name.lower():
                rwkv_grad_norms.append(param.grad.norm().item())
            elif 'mamba' in name.lower():
                mamba_grad_norms.append(param.grad.norm().item())
    
    rwkv_avg = sum(rwkv_grad_norms) / len(rwkv_grad_norms) if rwkv_grad_norms else 0
    mamba_avg = sum(mamba_grad_norms) / len(mamba_grad_norms) if mamba_grad_norms else 0
    ratio = rwkv_avg / (mamba_avg + 1e-9)
    
    return {'rwkv': rwkv_avg, 'mamba': mamba_avg, 'ratio': ratio}
```

**Acceptance Criteria:**
- [ ] Logging function implemented
- [ ] Ratio printed every 100 steps
- [ ] RED FLAG if ratio <0.1 or >10

---

### Task 7: Train 1K Steps, Pass Gate G3-G4-G4

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Quick training sanity check with component balance

**G3 Criteria:**
- [ ] Loss is decreasing
- [ ] Gradient norm 0.5-1.5 (warn if 1.5-3.0, FAIL if >5.0)
- [ ] Both train AND val loss logged

**G4 Criteria (Component Balance):**

| Metric | Pass | Warn | Fail |
|--------|------|------|------|
| Gradient Ratio (RWKV/Mamba) | 0.3-3.0 | 0.1-0.3 or 3-10 | <0.1 or >10 |

If G4 FAIL: One component is dead. Stop and investigate.

---

### Task 8: Run State Health Diagnostic (Gate G3.5)

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Verify state is not frozen or chaotic

**From V3 Section 9.5:**

| Metric | Method | Pass | Warn | Fail |
|--------|--------|------|------|------|
| State Evolution | Cosine similarity | < 0.99 | 0.95-0.99 | > 0.99 (frozen) |
| SVD Rank | Top-5 ratio | > 0.5 | 0.3-0.5 | < 0.3 |
| Gate Saturation | % values > 5.0 | < 10% | 10-30% | > 30% |

---

### Task 9: Train First Hybrid (100K Steps)

**Status:** ⬜ PENDING  
**Time:** ~2-4 hours runtime  
**Scope:** Full training run with monitoring

**Requirements:**
- Use dataset from Task 2
- Use config from Task 3
- Log every step: train loss
- Log every 100 steps: val loss, grad norms, component ratio
- Save checkpoints every 10K steps
- Monitor for stopping criteria

**Metrics Table:**

| Model | Steps | Final Train Loss | Final Val Loss | RWKV/Mamba Ratio | Memory | Tok/s |
|-------|-------|------------------|----------------|------------------|--------|-------|
| Hybrid-Balanced | 100K | ? | ? | ? | ? | ? |

**Acceptance Criteria:**
- [ ] Training completes OR stopping criteria triggered
- [ ] Both train AND val loss logged
- [ ] No stopping criteria triggered = Phase 1 PASS
- [ ] Document in V4_BUILD_LOG.md

---

## Models Reference

**All models use ParallelHybridBlock (RWKV6 + Mamba2 in every block)**

| Model | Layers | Hidden | Approx Params |
|-------|--------|--------|---------------|
| Hybrid-5M | 8 | 128 | ~5.2M |
| Hybrid-6M | 10 | 160 | ~6.5M |
| Hybrid-8M | 12 | 192 | ~8.0M |

**Phase 2 experiments:** Vary the rwkv_gain/mamba_gain initialization or ratio.  
**Phase 3 experiments:** Test different fusion mechanisms (concat, gated, etc).

**Architecture:** Parallel Hybrid Blocks (see V4_DESIGN.md)

---

## Key Diagnostic: Component Gradient Logging

```python
def log_component_gradients(model):
    """
    For Sequential Sandwich architecture.
    RWKV6 params are in layers 0 and 22.
    Mamba2 params are in layers 1-21.
    """
    rwkv_grad_norms = []
    mamba_grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Check layer type by parameter name
            if 'rwkv' in name.lower():
                rwkv_grad_norms.append(param.grad.norm().item())
            elif 'mamba' in name.lower():
                mamba_grad_norms.append(param.grad.norm().item())
    
    rwkv_avg = sum(rwkv_grad_norms) / len(rwkv_grad_norms) if rwkv_grad_norms else 0
    mamba_avg = sum(mamba_grad_norms) / len(mamba_grad_norms) if mamba_grad_norms else 0
    ratio = rwkv_avg / (mamba_avg + 1e-9)
    
    # Gate G4 check
    if ratio < 0.1 or ratio > 10:
        print(f"⚠️ GATE G4 FAIL: Gradient ratio {ratio:.2f} - one component may be dead!")
    elif ratio < 0.3 or ratio > 3:
        print(f"⚠️ GATE G4 WARN: Gradient ratio {ratio:.2f}")
    
    return {'rwkv': rwkv_avg, 'mamba': mamba_avg, 'ratio': ratio}
```

---

*Start with Task 1. Do not skip ahead.*
