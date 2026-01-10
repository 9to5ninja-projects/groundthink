# V4 Strategy Document - Task Backlog

**Created:** 2026-01-08  
**Updated:** 2026-01-10  
**Repository:** https://github.com/9to5ninja-projects/groundthink  
**Purpose:** Ordered queue of tasks to be completed one at a time  
**Current Goal:** Build RWKV-6 + Mamba-2 hybrid at 5M scale, find best configuration

---

## Scaling Philosophy & Confidence Criteria

**Foundation:** See [SCALING_MILESTONES.md](SCALING_MILESTONES.md) for strategic framework on what each scale should achieve and confidence criteria for graduation.

**Quick Summary:**
- **3.5M:** Sanity check (can training system work?)
- **8M:** Proof of concept (can architecture learn real patterns?)
- **30M:** Scaling laws (do predictions hold? do capabilities emerge?)
- **125M:** MVP delivery (is this production-ready?)

Each scale is a distinct experimental regime with specific learning objectives. Graduating to the next scale requires meeting technical, scientific, capability, and forecasting confidence criteria.

---

## Complexity Ratings & Procedures

**Tasks are rated S/M/L/XL based on scope and time.** For detailed assessment matrix and SOP self-improvement guidance, see [GETTING_STARTED.md](GETTING_STARTED.md#-task-complexity-assessment)

**Quick Reference:**
- **S** (Small): <30 min, 1 file, clear steps
- **M** (Medium): 30m-2h, 2-3 files, some research
- **L** (Large): 2-6h, 3-5 files, implementation needed
- **XL** (Extra Large): >6h, many files — **MUST break into S/M/L subtasks before starting**

---

## Librarian Role (Document Curator)

**Responsibilities by Documentation Scope:**

**Tier 1 (Always Required):**
- Assess task complexity (S/M/L/XL) and time estimates
- Track task completion status and phase transitions
- Update [V4_HANDOFF.md](V4_HANDOFF.md) with version/phase snapshots

**Tier 2 (At >50K documents):**
- Maintain navigation maps ([DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md))
- Recommend document splits (when >1000 lines)
- Audit cross-references for accuracy
- Flag ambiguous or temporal language

**Tier 3 (At >100K documents):**
- Enforce consistency across strategy/implementation/test layers
- Verify documents addressing same topic distinguish their scope
- Standardize language and formatting

**Tier 4 (Before Scaling Decisions):**
- Verify all planning documents updated for new phase
- Cross-check decision matrices across related docs
- Confirm success criteria are measurable and aligned

**Current Level:** Tier 2-3 (Phase 3.9 planning, >140K docs)

**Curated Artifacts:**
- [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md) — Navigation by layer
- [LIBRARIAN_REVIEW.md](LIBRARIAN_REVIEW.md) — Audit reports
- [V4_HANDOFF.md](V4_HANDOFF.md) — Version snapshots

---

## Architecture Notes

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

## Validation & Training Reference

**For detailed gate procedures (G1-G4), stopping criteria, loss monitoring, and validation protocols, see [V4_TESTING.md](V4_TESTING.md) and [V4_TRAINING_GUIDE.md](V4_TRAINING_GUIDE.md).**

**Quick Gate Reference:**
- **G1:** Forward pass (no NaN, correct shapes)
- **G2:** Init entropy (2.0-5.0 at step 0)
- **G3:** Train 1K steps (loss decreasing, grad 0.5-1.5)
- **G4:** Component balance (ratio 0.3-3.0)

---

## Using This Backlog

**This is the ordered task queue — check [V4_HANDOFF.md](V4_HANDOFF.md) for current phase and next task.** For workflow procedures and usage guidelines, see [GETTING_STARTED.md](GETTING_STARTED.md#-mandatory-workflow).

**Quick start:**
1. Check handoff for current phase
2. Pick NEXT incomplete task below (in order)
3. Use `manage_todo_list` tool to break down (required for L/XL tasks)
4. Work ONE task at a time
5. Mark complete and update handoff when done

---

## Task Backlog (Ordered by Phase)

### Phase 0: CUDA Kernel Integration (PRIORITY)

**Goal:** Integrate production CUDA kernels for 10-100x performance improvement before extensive training.

**Status:** ✅ COMPLETE (2026-01-09, Build Session 8)

**Strategy: Smart Modified Approach**

Mamba-2 uses mamba-ssm native CUDA kernels. RWKV-6 uses prototype + CUDA wrapper with JIT compilation.

| # | Task | Status | Completion | Dependencies | Complexity |
|---|------|--------|------------|--------------|------------|
| 0.1 | Mamba-2 CUDA kernels | ✅ COMPLETE | Pre-installed via mamba-ssm | - | S |
| 0.2 | RWKV-6 Prototype | ✅ COMPLETE | rwkv6_prototype.py, G1 passed | - | M |
| 0.3 | RWKV-6 CUDA Wrapper | ✅ COMPLETE | rwkv6_cuda_wrapper.py integrated | 0.2 | L |
| 0.4 | Hybrid Block Integration | ✅ COMPLETE | fla_replacements.py, hybrid_v4.py | 0.3 | M |

**Test Results (Full Gate Validation):**

| Gate | Test | Result | Value | Status |
|------|------|--------|-------|--------|
| G0 | Kernel availability | causal-conv1d, selective_scan, wkv6 | 3/3 | ✅ PASS |
| G1 | Forward pass | No NaN, correct shapes [2,64,256] | - | ✅ PASS |
| G2 | Init entropy | Logits at last token, step 0 | 5.46 | ⚠️ WARN |
| G3 | Mini training (100 steps) | Loss decrease, grad health | 1.08 | ✅ PASS |
| G3.5 | State evolution | Activation variance > 1e-6 | 3.5e-5 | ✅ PASS |
| G4 | Component balance | RWKV/Mamba grad ratio | 1.81 | ✅ PASS |

**Detailed Findings:**

**CUDA Kernels (G0):**
- causal_conv1d: ✓ Working (mamba-ssm native)
- selective_scan: ✓ Working (mamba-ssm native, verified in G3)
- wkv6: ✓ JIT compiles on first use (h32: 70 regs forward, 80 backward)

**Architecture Correctness (G1, Test 1-4):**
- RWKV-6 output: ✓ Shape [2,32,128], std 1.66, no NaN
- Mamba-2 output: ✓ Shape [2,32,128], std 0.58, no NaN
- Component independence: ✓ Cosine similarity -0.01 (completely independent)
- Gradient flow: ✓ Both components backpropagate correctly

**Initialization (G2):**
- Value: 5.46 (threshold 2.0-5.0, WARN 5.0-7.0)
- Status: ⚠️ Above pass threshold but in acceptable warn range
- Interpretation: Slightly higher uniform distribution than ideal, typical for hybrid
- Action: Monitor during training but not a blocker

**Training Dynamics (G3, 100-step mini run):**
- Initial loss: 5.5928
- Final loss: 5.5698 (↓ 0.023)
- Avg gradient norm: 1.0834 (pass range 0.5-1.5)
- Status: ✓ Loss decreasing with healthy gradients

**Component Balance (G4):**
- RWKV avg gradient: 0.00821 (112 params)
- Mamba avg gradient: 0.00453 (80 params)
- Ratio: 1.81 (pass range 0.3-3.0)
- Status: ✓ Well-balanced, RWKV slightly stronger but within bounds

**State Health (G3.5):**
- Activation variance (RWKV): 3.5e-5
- Activation variance (Mamba): 3.6e-5
- Status: ✓ Both varying with input, not frozen

**Files Created/Modified:**

**New:**
- `test_phase0_complete.py` - Comprehensive validation (7 tests + 5 gates)

**Modified:**
- `fla_replacements.py` - CUDA wrapper integration with fallback
- `rwkv6_cuda_wrapper.py` - Compiler env vars fixed

**Unchanged (Verified Working):**
- `rwkv6_prototype.py`, `hybrid_v4.py`, `data_v030.py`

**Cross-References:**
- Build log: [V4_BUILD_LOG.md - Session 8](V4_BUILD_LOG.md#build-session-8-2026-01-09)
- Test file: [test_phase0_complete.py](test_phase0_complete.py)
- Kernel specs: [V4.5_CUDA_KERNELS.md](V4.5_CUDA_KERNELS.md)
- Testing guide: [V4_TESTING.md](V4_TESTING.md)

---

### Phase 1: Foundation (Must Complete First)

| # | Task | Status | Depends On | Gates | Complexity | Source/Details |
|---|------|--------|------------|-------|------------|----------------|
| 1 | ~~Verify data_v030.py~~ | ✅ COMPLETE | - | - | S | Archived, train_v4.py uses it |
| 2 | ~~Validate Test Dataset~~ | ✅ COMPLETE | Task 1 | - | S | shakespeare.txt, 1.1M tokens |
| 3 | Build First Hybrid (ParallelHybridBlock) | ✅ COMPLETE | - | G1 | L | [V4_DESIGN.md](V4_DESIGN.md) - hybrid_v4.py (3.8M) |
| 4 | ~~Define Training Configuration~~ | ✅ COMPLETE | Task 3 | - | M | train_v4.py created |
| 5 | ~~First Training Run~~ | ✅ COMPLETE | Task 4 | G1-G2, G3 | L | 5K steps, Loss 1.37, G4 WARN |
| 6 | ~~Setup Performance Monitoring~~ | ✅ COMPLETE | Task 5 | - | S | [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) Phase 1 |
| 6.5 | ~~Test Monitoring Tools~~ | ✅ COMPLETE | Task 6, 0.2 | - | S | Completed via test_phase0_complete.py |
| 6.6 | ~~Research RWKV-6 Architecture~~ | ✅ SUPERSEDED | Task 6.5 | - | M | SUPERSEDED by Task 0.1 (direct kernel integration) |
| 6.7 | ~~Research Mamba-2 Architecture~~ | ✅ SUPERSEDED | Task 6.6 | - | M | SUPERSEDED by Task 0.2 (official kernels) |
| 6.8 | ~~Audit Custom Wrappers~~ | ✅ SUPERSEDED | Tasks 6.6, 6.7 | - | M | SUPERSEDED by Phase 0 |
| 6.9 | ~~Implement/Fix RWKV-6 Component~~ | ✅ SUPERSEDED | Task 6.8 | G1 | L/XL | SUPERSEDED by Task 0.1 |
| 6.10 | ~~Implement/Fix Mamba-2 Component~~ | ✅ SUPERSEDED | Task 6.8 | G1 | L/XL | SUPERSEDED by Task 0.2 |
| 6.11 | ~~Rebuild hybrid_v4.py~~ | ✅ COMPLETE | Tasks 0.1, 0.2 | G1-G2 | M | fla_replacements.py integrates CUDA wrappers |
| 6.12 | ~~Verify Model Works~~ | ✅ COMPLETE | Task 6.11 | G1-G2 | M | test_phase0_complete.py - all gates pass |
| 7 | ~~Baseline Performance Profiling~~ | ✅ COMPLETE | Task 6.12 | - | M | benchmark_suite.py, Session 9 |
| 8 | ~~Apply Quick Win Optimizations~~ | ✅ COMPLETE | Task 7 | - | L | Session 10: 186K tok/s (+586% vs baseline) |
| 9 | ~~Run Controlled Experiments~~ | ✅ COMPLETE | Task 8 | - | XL | Tested batch scaling + AMP combinations |
| 10 | ~~Select Optimal Configuration~~ | ✅ COMPLETE | Task 9 | - | M | batch=64, AMP=True, mamba_lr_mult=0.5 |
| 11 | ~~Analyze Training Results~~ | ✅ COMPLETE | Task 10 | G3.5, G4 | M | Ratio stabilized at 0.7-1.3 |
| 12 | ~~Address Gradient Imbalance~~ | ✅ COMPLETE | Task 11 | G4 (fix) | L | mamba_lr_mult: 2.0→0.5 fixed G4 |
| 13 | ~~Extended Training Run~~ | ✅ COMPLETE | Task 12 | G1-G4 | L | 5K steps, loss 4.60→1.14 (-75%) |

**Current Status:** Phase 1 COMPLETE (2026-01-09 Session 11). Phase 2 (Tasks 14+) is **NEXT**.

**Task 13 Results (5000 steps):**
- Loss: 4.60 → 1.14 train, 1.49 val (**-75% reduction**)
- PPL: 92.5 → 3.12 (**-97% reduction**)
- Throughput: 35K tok/s avg (40K peak)
- Duration: 582.4s (~9.7 min)
- Tokens: 20.48M processed
- G1-G4: All passed (G4 drifted to 0.29 at low LR, expected)

**Optimized Metrics (Session 10):**
- B1 Throughput: 186,398 tok/s (+586% vs baseline)
- B2 Peak VRAM: 184.9 MiB (still under 200 MiB target)
- B3 Stability: PASS (loss 9.17→2.51 in 200 steps)
- G4 Gradient Ratio: 0.7-1.3 ✅ PASS (was 0.15-0.16)

**Key Fixes Applied:**
- Batch size: 8 → 64 (6.1x throughput)
- AMP enabled: +12% throughput
- mamba_lr_mult: 2.0 → 0.5 (gradient balance fixed)

---

### Phase 2: Fusion & Ratio Comparison ✅ COMPLETE

**Goal:** Find the standout winner among fusion strategies and ratio variants.

#### Task 14 Results: Fusion Benchmark (2026-01-09)

| Rank | Code | Val Loss | Train Loss | PPL | Throughput |
|------|------|----------|------------|-----|------------|
| 🥇 | **GF** | **1.6891** | 1.6536 | 5.41 | 42.9K tok/s |
| 🥈 | CP | 1.6919 | 1.6544 | 5.43 | 47.7K tok/s |
| 🥉 | HY | 1.7600 | 1.7289 | 5.81 | 31.7K tok/s |
| 4 | WS | 1.8185 | 1.7935 | 6.16 | 45.4K tok/s |
| 5 | RF | 1.9480 | 1.9339 | 7.01 | 47.4K tok/s |

**Winner: GF (Gated Fusion)** - 4% better than HY baseline, 35% faster.

#### Task 15-17 Results: Ratio Benchmark (2026-01-09)

| Rank | Code | Val Loss | Train Loss | PPL | Throughput |
|------|------|----------|------------|-----|------------|
| 🥇 | **GF-MH** | **1.6700** | 1.6321 | 5.31 | 47.2K tok/s |
| 🥈 | GF | 1.6998 | 1.6604 | 5.47 | 36.8K tok/s |
| 🥉 | GF-RH | 1.7201 | 1.6866 | 5.59 | 44.0K tok/s |

**Winner: GF-MH (Mamba-Heavy)** - Gate init 0.3 favoring Mamba, 2% better than balanced GF.

#### All Variants Summary

| File | Code | Description | Val Loss | Status |
|------|------|-------------|----------|--------|
| hybrid_v4_ratio.py | **GF-MH** | GF + Mamba-Heavy | **1.6700** | 🏆 **OVERALL WINNER** |
| hybrid_v4_GF.py | GF | Gated Fusion balanced | 1.6998 | ✅ Fusion winner |
| hybrid_v4_ratio.py | GF-RH | GF + RWKV-Heavy | 1.7201 | ✅ Tested |
| hybrid_v4_CP.py | CP | Concat+Project | 1.6919 | ✅ Close to GF |
| hybrid_v4.py | HY | Per-channel gains | 1.7600 | ✅ Baseline |
| hybrid_v4_WS.py | WS | Weighted Sum | 1.8185 | ✅ Tested |
| hybrid_v4_RF.py | RF | Residual Fusion | 1.9480 | ✅ Worst |

#### Phase 2 Task Table

| # | Task | Status | Depends On | Complexity | Details |
|---|------|--------|------------|------------|---------|
| 14 | ~~Benchmark Fusion Variants~~ | ✅ COMPLETE | Task 13 | M | GF wins fusion |
| 15 | ~~Build RWKV-Heavy Hybrid~~ | ✅ COMPLETE | Task 14 | M | GF-RH in hybrid_v4_ratio.py |
| 16 | ~~Build Mamba-Heavy Hybrid~~ | ✅ COMPLETE | Task 14 | M | GF-MH in hybrid_v4_ratio.py |
| 17 | ~~Benchmark All 3 Ratio Variants~~ | ✅ COMPLETE | Tasks 15, 16 | L | GF-MH wins ratio |
| 18 | ~~Select Winner & Document~~ | ✅ COMPLETE | Task 17 | S | **GF-MH is final winner** |

**Phase 2 Gate: PASSED** - GF-MH (Gated Fusion + Mamba-Heavy) is the winner.

**Key Insight:** Mamba benefits from higher relative weight in the hybrid. RWKV-Heavy performs worst.

---

### Phase 2.5: Infrastructure & Evaluation (Before Extended Training)

**Rationale:** Rushing to train without proper infrastructure leads to tedious manual edits and no way to evaluate quality. Build once, use forever.

| # | Task | Status | Depends On | Complexity | Details |
|---|------|--------|------------|------------|---------|
| 18.1 | Model Registry & Factory | ✅ COMPLETE | Task 18 | M | models/__init__.py, --model CLI arg |
| 18.2 | Centralized Config System | ✅ COMPLETE | Task 18.1 | M | YAML configs + CLI overrides working |
| 18.3 | NIAH Test Implementation | ⬜ **NEXT** | Task 18.1 | L | Test existing 5M model first |
| 18.4 | Qualitative Eval Suite | ⬜ PENDING | Task 18.3 | M | Generation samples, perplexity, patterns |
| 18.5 | Evaluation Baseline | ⬜ PENDING | Task 18.4 | M | Run eval suite on 5M winner (GF-MH) |

**Gate:** Phase 2.5 complete when we can switch models via CLI and have baseline eval metrics.

---

#### Task 18.1: Model Registry & Factory

**Status:** ⬜ PENDING  
**Complexity:** M (Medium)  
**Time:** ~45 min  
**Priority:** 🔴 HIGH - Fixes the import pain permanently

**Problem:** 
Every time we switch models, we edit imports in train_v4.py. This is error-prone and tedious.

**Solution:** Create `models/__init__.py` with a registry pattern:

```python
# models/__init__.py
from .hybrid_v4 import create_hybrid_5m
from .hybrid_v4_8m import create_hybrid_8m
from .hybrid_v4_GF import create_hybrid_GF
from .hybrid_v4_ratio import create_hybrid_GF_MH, create_hybrid_GF_RH

REGISTRY = {
    '5M': create_hybrid_5m,
    '8M': create_hybrid_8m,
    'GF': create_hybrid_GF,
    'GF-MH': create_hybrid_GF_MH,
    'GF-RH': create_hybrid_GF_RH,
}

def get_model(name: str, vocab_size: int, **kwargs):
    """Get model by name. Usage: model = get_model('8M', vocab_size=97)"""
    if name not in REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](vocab_size=vocab_size, **kwargs)
```

**Usage in train_v4.py:**
```python
from models import get_model
model = get_model(args.model, vocab_size=tokenizer.vocab_size)
```

**Command line:**
```bash
python train_v4.py --model 8M --steps 50000
python train_v4.py --model GF-MH --steps 5000
```

**Acceptance Criteria:**
- [ ] `models/` directory created with `__init__.py`
- [ ] All model variants moved to `models/` or imported
- [ ] `get_model(name)` function works for all variants
- [ ] train_v4.py uses `--model` argument
- [ ] No more hardcoded imports to change

---

#### Task 18.2: Centralized Config System

**Status:** ⬜ PENDING  
**Complexity:** M (Medium)  
**Time:** ~30 min  
**Priority:** 🟠 HIGH - Reduces config scatter

**Problem:**
Config values are hardcoded in train_v4.py. Changing them requires editing the file.

**Solution:** YAML config files + CLI overrides:

```yaml
# configs/train_8m_50k.yaml
model: 8M
max_steps: 50000
warmup_steps: 2500
batch_size: 32
grad_accum: 4
seq_len: 128
lr: 3e-4
min_lr: 3e-5
use_amp: true
eval_every: 500
save_every: 5000
```

**Usage:**
```bash
python train_v4.py --config configs/train_8m_50k.yaml
python train_v4.py --config configs/train_8m_50k.yaml --lr 1e-4  # Override
```

**Acceptance Criteria:**
- [ ] `configs/` directory with preset configs
- [ ] Config loader in train_v4.py
- [ ] CLI args override config file values
- [ ] Default config for quick testing

---

#### Task 18.3: NIAH Test Implementation

**Status:** ⬜ PENDING  
**Complexity:** L (Large)  
**Time:** ~2 hours  
**Priority:** 🟠 HIGH - Quality validation before extended training

**What is NIAH?**
Needle-in-a-Haystack: Hide a fact in filler text, query model to retrieve it. Tests long-context memory.

**Why test existing 5M model FIRST?**
- We have trained checkpoints already (ckpt_HY_step5000.pt, etc.)
- Establishes baseline before scaling
- May reveal issues that more training won't fix

**Simple Implementation (no external deps):**

```python
# eval/niah.py
def niah_test(model, tokenizer, context_length=1024, needle_position=0.5):
    """
    Test long-context retrieval.
    
    Args:
        context_length: Total tokens of context
        needle_position: Where to hide needle (0.0=start, 0.5=middle, 1.0=end)
    """
    # Needle (fact to remember)
    needle = "The secret code is ALPHA-7."
    query = "What is the secret code?"
    
    # Haystack (filler text from shakespeare.txt)
    haystack = load_filler_text(context_length - len(tokenizer.encode(needle)))
    
    # Insert needle at position
    insert_idx = int(len(haystack) * needle_position)
    context = haystack[:insert_idx] + needle + haystack[insert_idx:]
    
    # Query model
    prompt = context + "\n\nQuestion: " + query + "\nAnswer:"
    response = model.generate(prompt, max_tokens=20)
    
    # Check if correct
    success = "ALPHA-7" in response
    return {'success': success, 'response': response, 'context_len': context_length}
```

**Test Matrix:**
| Context | Position | Expected |
|---------|----------|----------|
| 512 | start | Should pass (easy) |
| 512 | middle | Should pass |
| 512 | end | Should pass |
| 1024 | middle | Baseline test |
| 2048 | middle | Stress test |
| 4096 | middle | May fail (context limit) |

**Acceptance Criteria:**
- [ ] `eval/niah.py` created
- [ ] Works with existing checkpoints
- [ ] Results logged to console/file
- [ ] Baseline numbers for 5M model documented

---

#### Task 18.4: Qualitative Eval Suite

**Status:** ⬜ PENDING  
**Complexity:** M (Medium)  
**Time:** ~45 min  
**Priority:** 🟡 MEDIUM - Nice to have before extended training

**What to Evaluate:**

1. **Text Generation Quality:**
   ```python
   prompts = [
       "To be or not to be,",
       "Once upon a time,",
       "The meaning of life is",
   ]
   for p in prompts:
       print(f"Prompt: {p}")
       print(f"Output: {model.generate(p, max_tokens=50)}\n")
   ```

2. **Perplexity on Held-Out Text:**
   ```python
   val_ppl = compute_perplexity(model, val_dataset)
   print(f"Validation PPL: {val_ppl:.2f}")
   ```

3. **Token Distribution:**
   - Are outputs diverse or repetitive?
   - Entropy of generated tokens

4. **Attention/State Patterns (optional):**
   - Visualize where model "attends"
   - State evolution over sequence

**Acceptance Criteria:**
- [ ] `eval/qualitative.py` created
- [ ] Generation, PPL, diversity metrics
- [ ] Works with any model from registry
- [ ] Output to console + optional file

---

#### Task 18.5: Evaluation Baseline

**Status:** ⬜ PENDING  
**Complexity:** M (Medium)  
**Time:** ~30 min  
**Priority:** 🟡 MEDIUM - Establishes comparison point

**Run eval suite on existing 5M GF-MH checkpoint:**

```bash
python eval/run_eval.py --model GF-MH --checkpoint ckpt_HY_step5000.pt
```

**Expected Output:**
```
=== NIAH Results ===
Context 512 (start):  PASS
Context 512 (middle): PASS  
Context 512 (end):    PASS
Context 1024 (middle): PASS
Context 2048 (middle): FAIL (degradation point)

=== Generation Samples ===
Prompt: "To be or not to be,"
Output: "that is the question. Whether 'tis nobler..."

=== Metrics ===
Val PPL: 4.44
Token Diversity: 0.73
```

**Acceptance Criteria:**
- [ ] Baseline numbers documented
- [ ] Clear degradation point identified (NIAH)
- [ ] Generation quality assessed (subjective but documented)
- [ ] Results in V4_BUILD_LOG.md

---

### Phase 3: Scale Testing (After Phase 2.5)

| # | Task | Status | Depends On | Complexity | Details |
|---|------|--------|------------|------------|---------|
| 19 | Scale GF-MH to 8M Params | ✅ COMPLETE | Task 18 | L | hybrid_v4_8m.py (7.93M) |
| 20 | Extended Training (50K steps) | ⬜ PENDING | Tasks 18.1, 18.2 | XL | `--model 8M --config train_8m_50k.yaml` |
| 21 | Post-Training Eval | ⬜ PENDING | Task 20 | M | Run eval suite on 8M model |

**Gate:** Phase 3 complete when 8M model trained and eval shows improvement over 5M baseline.

**Note:** Tasks 19.1-19.5 (model import fixes, config updates) are **SUPERSEDED** by Phase 2.5 infrastructure (registry + config system). Once 18.1 and 18.2 are done, training any model is just:

```bash
python train_v4.py --model 8M --config configs/train_8m_50k.yaml
```

No more manual import edits. No more scattered config changes.

---

#### Task 19.1: Fix train_v4.py Model Import

**Status:** ⬜ PENDING  
**Complexity:** S (Small)  
**Time:** ~5 min  
**Priority:** 🔴 BLOCKER - Cannot run 8M training without this

**Problem:** 
`train_v4.py` line 15 imports `create_hybrid_5m` from `hybrid_v4.py`, but we need the 8M model.

**Current:**
```python
from hybrid_v4 import create_hybrid_5m
```

**Required:**
```python
from hybrid_v4_8m import create_hybrid_8m
```

**Also need to update:**
- Line ~259: `model = create_hybrid_5m(...)` → `model = create_hybrid_8m(...)`

**Acceptance Criteria:**
- [ ] Import changed to hybrid_v4_8m
- [ ] Model creation uses create_hybrid_8m
- [ ] Script runs without import errors

---

#### Task 19 Results: 8M Model Benchmark (2026-01-10)

**Hardware:** RTX 4050 Laptop GPU (6GB VRAM)

| Config | Throughput | VRAM | Notes |
|--------|------------|------|-------|
| FP32 baseline | 15.6K tok/s | 1,336 MiB | batch=32, seq=128 |
| FP16 AMP | 17.9K tok/s | 1,111 MiB | +15% speed, -17% VRAM |
| BF16 AMP | 4.5K tok/s | 1,119 MiB | ❌ Slow (no native kernel) |
| batch=48 + FP16 | 47.4K tok/s | 1,336 MiB | Best raw throughput |
| **batch=32 + grad_accum=4 + FP16** | **45.1K tok/s** | **972 MiB** | ✅ **RECOMMENDED** |

**Task 20 Recommended Config (for configs/train_8m_50k.yaml):**
```yaml
model: 8M
batch_size: 32
grad_accum: 4
use_amp: true
max_steps: 50000
warmup_steps: 2500
# Effective batch: 128, VRAM: ~1GB
# 50K steps ETA: ~75 min at 45K tok/s
```

---

### Phase 3.5: Parallel Training & Architecture Tuning

**Rationale:** GPU analysis shows RTX 4050 (6GB) can fit 2x small models at batch=32 using only 47% VRAM. Multi-worker training can accelerate experiments.

| # | Task | Status | Depends On | Complexity | Details |
|---|------|--------|------------|------------|---------|
| 22 | GPU VRAM Analysis | ✅ COMPLETE | - | S | Documented: 2x small @ batch=32 = 2.9GB (47%) |
| 23 | Multi-Worker Training Script | ⬜ **NEXT** | Task 22 | M | Run 2+ models in parallel for A/B comparisons |
| 24 | Architecture Tuning Guide | ⬜ PENDING | Task 22 | M | Document tunable params: gate_init, mamba_lr_mult, etc. |
| 24.1 | HGF Balance Tuning | ⬜ PENDING | Task 24 | L | Fix RWKV-dominance in HGF (gradient ratio 0.22) |

**Gate:** Phase 3.5 complete when we can run parallel experiments and have documented tuning parameters.

---

#### Task 22: GPU VRAM Analysis

**Status:** ✅ COMPLETE (2026-01-10)  
**Complexity:** S (Small)  
**Time:** ~15 min  

**Results (RTX 4050 Laptop, 6GB VRAM):**

| Model | Batch | VRAM (train) | Headroom | 2x Workers? |
|-------|-------|--------------|----------|-------------|
| tiny | 32 | 424MB | 5.7GB | ✅ Yes |
| tiny | 64 | 826MB | 5.3GB | ✅ Yes |
| small | 32 | 1.5GB | 4.7GB | ✅ Yes (2x = 2.9GB) |
| small | 64 | 2.9GB | 3.2GB | ⚠️ Tight |
| HGF | 32 | 1.5GB | 4.7GB | ✅ Yes |
| medium | 32 | 2.1GB | 4.0GB | ⚠️ Maybe 2x at batch=16 |

**Key Finding:** 2x small models at batch=32 uses only **47% VRAM** (2.9GB). Headroom: 3.2GB.

**Safe Multi-Worker Configs:**
- 2x tiny, batch=64 → ~1.6GB ✅
- 2x small, batch=32 → ~2.9GB ✅
- 2x small, batch=48 → ~4.2GB ✅
- 2x HGF, batch=32 → ~3.0GB ✅
- 2x medium, batch=16 → ~2.6GB ✅

---

#### Task 23: Multi-Worker Training Script

**Status:** ⬜ PENDING  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Priority:** 🟡 MEDIUM - Accelerates A/B experiments

**Goal:** Create script to run 2+ models in parallel on single GPU for:
- Comparative experiments (e.g., HGF vs GF-MH)
- Architecture ablations
- Hyperparameter sweeps

**Approach Options:**
1. **Sequential interleaved** — Alternate batches between models
2. **True parallel** — Both models forward/backward same batch, aggregate gradients
3. **Async workers** — Python multiprocessing with shared GPU memory

**Acceptance Criteria:**
- [ ] Script runs 2 models simultaneously
- [ ] VRAM stays within safe limits (< 5GB for small)
- [ ] Results logged separately per model
- [ ] Supports different model variants (e.g., HGF vs GF-MH)

---

#### Task 24: Architecture Tuning Guide

**Status:** ⬜ PENDING  
**Complexity:** M (Medium)  
**Time:** ~45 min  
**Priority:** 🟡 MEDIUM - Enables systematic experiments

**Goal:** Document all tunable architecture/training parameters for component balance:

**Known Tuning Parameters:**

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `gate_init` | Model constructor | 0.5 | Initial RWKV/Mamba blend (0=Mamba, 1=RWKV) |
| `mamba_lr_mult` | train_v4.py | 0.5 | Mamba learning rate multiplier |
| `rwkv_lr_mult` | train_v4.py | 1.0 | RWKV learning rate multiplier |
| `ffn_mult` | Model constructor | 4.0 | FFN hidden size ratio |
| `num_heads` | Model constructor | 4 | Attention heads (affects both RWKV & Mamba) |

**Acceptance Criteria:**
- [ ] All tunable params documented with effects
- [ ] Recommended ranges for balance
- [ ] Guide added to V4_TRAINING_GUIDE.md

---

#### Task 24.1: HGF Balance Tuning

**Status:** ⬜ PENDING  
**Complexity:** L (Large)  
**Time:** ~2 hours  
**Priority:** 🟡 MEDIUM - HGF shows promise but needs tuning

**Problem:** HGF validation showed RWKV dominance:
- Gradient ratio: 0.22 (target: 0.3-3.0)
- Activation variance ratio: 47x (RWKV >> Mamba)

**Hypothesis:** GF-MH fixed this with `mamba_lr_mult=0.5`. HGF may need:
- Lower `gate_init` (e.g., 0.3 to favor Mamba initially)
- Different `mamba_lr_mult`
- Per-layer gate initialization

**Experiment Plan:**
1. HGF with `gate_init=0.3` (Mamba-heavy)
2. HGF with `mamba_lr_mult=0.3`
3. Compare gradient ratios after 500 steps

**Acceptance Criteria:**
- [ ] Find config that achieves gradient ratio 0.3-3.0
- [ ] Document winning config
- [ ] Update HGF model defaults if needed

---

### Phase 3.6: Pre-Training Benchmark Suite ("Training for Training") ⚠️ CHAR-LEVEL ONLY

**⚠️ WARNING:** All Phase 3.6 experiments used **char-level tokenization** (Shakespeare). This was appropriate for quick sanity checks but does NOT represent production conditions. Results must be re-validated with BPE before use. See Phase 4.0 for re-validation plan.

**Rationale:** Before expensive training runs, we test fusion variants on cheap diagnostic benchmarks. This finds optimal "brain mix" without wasting GPU hours.

**Key Insight (Session 13):** 
- NIAH retrieval tests don't work for char-level models (they predict, not retrieve)
- LRD (Long-Range Dependency) tests show context utilization via perplexity improvement
- Fusion variants can be compared on untrained models to find best architecture

| # | Task | Status | Depends On | Complexity | Details |
|---|------|--------|------------|------------|---------|
| 25 | LRD Test Script | ✅ COMPLETE | - | S | tests/test_lrd.py - measures context benefit |
| 26 | Fusion Variant LRD Comparison | ✅ COMPLETE | Task 25 | M | Untrained: ~0% all variants (expected) |
| 27 | NIAH Test Script | ⚠️ DEPRIORITIZED | - | S | Not valid for char-level models |
| 28 | Quick Train Comparison | ✅ COMPLETE | Task 26 | M | 1K-step training on 5 variants |
| 29 | Component Balance Tests | ✅ COMPLETE | Task 28 | M | Results in V4_FUSION_MODELS.md |
| 30 | Fusion Variant Ranking | ✅ COMPLETE | Task 29 | S | See Phase 3.6 Results below |
| 30.1 | Fix Checkpoint Naming | ✅ COMPLETE | - | S | train_v4.py now uses ckpt_{MODEL}_*.pt |

**Gate:** ✅ Phase 3.6 COMPLETE (2026-01-10)

**Results Summary:**
| Variant | Val Loss | R/M Ratio | Verdict |
|---------|----------|-----------|---------|
| GF-MH | **1.59** | 0.10 ⚠️ | Best loss, RWKV dominant |
| GF | 1.61 | 0.12 ⚠️ | Good loss, RWKV dominant |
| CP | 1.61 | 0.19 ⚠️ | Good loss, RWKV dominant |
| HGF | 1.69 | 0.21 ⚠️ | Mid loss, RWKV dominant |
| HY | 1.69 | **0.45** ✅ | Mid loss, balanced |

**Key Insight:** Position-adaptive fusion (GF variants) achieves lower loss but causes Mamba underutilization. HY's fixed per-dimension blend maintains component balance.

**Next Steps:**
1. For quick experiments: Use GF-MH (best loss)
2. For balanced training: Use HY (best gradient ratio)
3. For extended testing: Try HGF with gate_init=0.3 (Mamba-heavy)

---

### Phase 3.7: Blend Ratio Sweep ✅ COMPLETE ⚠️ CHAR-LEVEL ONLY

**⚠️ WARNING:** All Phase 3.7 experiments used **char-level tokenization**. The "RWKV dominance" finding may be a char-level artifact. Re-validation with BPE is required. See Phase 4.0.

**Objective:** Determine if RWKV dominance is architectural or signal-based.

**Documentation:** [V4_BLEND_RATIOS.md](V4_BLEND_RATIOS.md) — Full results and analysis

**Context:** Phase 3.6 showed ALL gated variants become RWKV-dominant (R/M ratio 0.10-0.21). We tested symmetric configurations to determine cause.

| # | Task | Status | Depends On | Complexity | Details |
|---|------|--------|------------|------------|---------|
| 31 | GF-RH 1K Training | ✅ COMPLETE | Phase 3.6 | S | Val Loss 1.64, R/M 0.14 |
| 32 | HGF-MH 1K Training | ✅ COMPLETE | Task 31 | S | Val Loss 1.69, R/M 0.24 |
| 33 | HGF-RH 1K Training | ✅ COMPLETE | Task 32 | S | Val Loss 1.70, R/M 0.25 |
| 34 | Gate Drift Analysis | ✅ COMPLETE | Tasks 31-33 | M | All variants → RWKV dominant |
| 35 | Phase 3.7 Decision | ✅ COMPLETE | Task 34 | S | **SIGNAL DOMINANCE CONFIRMED** |

**Results:**
| Model | gate_init | Final R/M | Val Loss | Val PPL |
|-------|-----------|-----------|----------|---------|
| GF-RH | 0.7 (RWKV) | 0.14 | 1.64 | 5.14 |
| HGF-MH | 0.3 (Mamba) | 0.24 | 1.69 | 5.40 |
| HGF-RH | 0.7 (RWKV) | 0.25 | 1.70 | 5.46 |

**Gate:** ✅ Phase 3.7 COMPLETE (2026-01-10)

**Conclusion: SIGNAL DOMINANCE CONFIRMED**

All gated variants converge to RWKV-dominant regardless of initial bias:
- GF-MH (started 30:70 Mamba) → R/M 0.10 (RWKV dominant)
- GF-RH (started 70:30 RWKV) → R/M 0.14 (RWKV dominant)
- HGF-MH (started 30:70 Mamba) → R/M 0.24 (RWKV dominant)
- HGF-RH (started 70:30 RWKV) → R/M 0.25 (RWKV dominant)

**Root Cause:** RWKV produces smoother gradients that are easier to optimize. The gate "takes the path of least resistance."

---

### Phase 3.7 Inferences (Data Analysis)

**Complete Results Table:**
| Model | Fusion Type | gate_init | R/M Ratio | Val Loss |
|-------|-------------|-----------|-----------|----------|
| GF-MH | Per-position gate | 0.3 | 0.10 | 1.59 |
| GF | Per-position gate | 0.5 | 0.12 | 1.61 |
| GF-RH | Per-position gate | 0.7 | 0.14 | 1.64 |
| CP | Learned projection | — | 0.19 | 1.61 |
| HGF | Per-pos+dim gate | 0.5 | 0.21 | 1.69 |
| HGF-MH | Per-pos+dim gate | 0.3 | 0.24 | 1.69 |
| HGF-RH | Per-pos+dim gate | 0.7 | 0.25 | 1.70 |
| HY | Fixed per-dim gains | — | 0.45 | 1.69 |

**Key Inferences:**

1. **Per-dimension gating preserves Mamba better:**
   - HGF variants: R/M 0.21-0.25 (~2x better than GF)
   - GF variants: R/M 0.10-0.14 (worst balance)
   - *Insight:* Dimension-level control prevents wholesale Mamba collapse

2. **Loss and balance are inversely correlated (Pareto frontier):**
   - Best loss (GF-MH 1.59) → Worst balance (R/M 0.10)
   - Best balance (HY 0.45) → Higher loss (1.69)
   - *Insight:* Can't have both without explicit intervention

3. **Interpolation semantics > free projection:**
   - HGF (33K params): R/M 0.21-0.25, constrained to [0,1] blend
   - CP (33K params): R/M 0.19, arbitrary linear combination
   - *Insight:* The sigmoid constraint in HGF helps balance

4. **HGF is the best gated variant for balance:**
   - HGF-MH: R/M 0.24, Loss 1.69 (same loss as HY, better than GF-MH's balance)
   - If balance matters, HGF-MH > GF-MH

**WS/RF not retested** — Phase 2 already showed too simple (loss 1.82-1.95)

---

### Phase 3.8: Balance Improvement Options

**Objective:** Improve Mamba utilization without sacrificing loss

| # | Task | Priority | Complexity | Status | Result |
|---|------|----------|------------|--------|--------|
| 36 | Increase Mamba LR (1.0x) | 🔴 HIGH | S | ✅ DONE | **WORSE** - R/M 0.10→0.08 |
| 37 | Differential warmup schedules | 🟠 MED | L | ⬜ | Deprioritized — requires re-evaluation |
| 38 | Balance regularization | 🟡 LOW | L | ⬜ | Deprioritized — requires re-evaluation |
| 39 | ~~Accept RWKV dominance~~ | — | — | ⚠️ INCONCLUSIVE | See Task 40 results |
| 40 | BPE benchmark validation | 🔴 HIGH | M | ✅ COMPLETE | See results below |

**Task 36: Increase Mamba LR (1.0x)** ✅ COMPLETE
- **Rationale:** Currently using 0.5x LR for Mamba. Mamba may be underfitting.
- **Command:** `mamba_lr_mult = 1.0`, ran GF-MH 1K steps
- **Result:** R/M **WORSE** (0.10→0.08), Val loss improved (1.59→1.53)
- **Conclusion:** Mamba is NOT LR-starved. Higher LR accelerates convergence to RWKV-dominant state.

**Task 37: Differential Warmup Schedules** ⚠️ DEPRIORITIZED — RESEARCH SHOWS TOKENIZATION IS ROOT CAUSE
- **Original idea:** Freeze gates for 500 steps
- **User insight:** RWKV and Mamba may have fundamentally different warmup requirements
- **Current state:** Single warmup schedule (500 steps) applied to ALL parameter groups equally

⚠️ **DECISION (2026-01-10):** Task 37 research was completed (see below) but is now **DEPRIORITIZED** based on V4.5_VALIDATION.md Entries V3-V4. Root cause analysis proved:
- Component imbalance is NOT a learning rate/warmup problem (Entry V3: higher Mamba LR worsens balance)
- Component imbalance IS a tokenization problem (Entry V4: BPE fixes balance to 0.20-0.46 vs char-level 0.08-0.11)
- **Action:** Use BPE tokenization instead of differential warmup (Task 40 BPE benchmark)
- **Status:** Research materials preserved below for reference; do NOT implement without re-evaluation

**Hypothesis** (Historical — preserved for reference):

If Mamba needs longer warmup:
- RWKV hits full LR at step 500
- Mamba still ramping → falls behind → gates lock to RWKV early

**Research Findings (2026-01-10):**

| Source | Component | Warmup Recommendation |
|--------|-----------|----------------------|
| BlinkDL/RWKV-LM | RWKV-6/7 | `warmup_steps=20` for spike prevention, 2000-2500 for large models |
| BlinkDL/RWKV-LM | State-tuning | `warmup_steps=10` with very high LR |
| BlinkDL/RWKV-LM | Spike fix | `lr * (0.01 + 0.99 * step / warmup)` (slower ramp) |
| state-spaces/mamba | Mamba-2 | No explicit guidance, follows GPT-3 conventions |
| state-spaces/mamba | SSMs | "Sensitive to recurrent dynamics" - needs higher precision |
| state-spaces/mamba | Δ param | Special initialization critical, may conflict with framework resets |

**Key Insight:** RWKV has specific warmup guidance (slow ramp helps stability). Mamba follows general transformer conventions but is "sensitive" and needs careful precision handling.

**Research Status:** ✅ COMPLETE (documented in [V4_TRAINING_GUIDE.md](V4_TRAINING_GUIDE.md#per-component-warmup-schedules-phase-38))

**Key Findings:**
- RWKV warmup: 20-2500 steps (BlinkDL spike-prevention formula available)
- Mamba warmup: No explicit guidance; SSMs sensitive to recurrent dynamics
- PyTorch LambdaLR: Supports per-group schedules natively via `lr_lambda=[lambda1, lambda2, lambda3]`

**Experiments (Detailed in Guide):**
- 37a: Mamba extended warmup (2x duration)
- 37b: RWKV slow ramp (BlinkDL spike-fix formula)
- 37c: Mamba delayed start
- 37d: Combined (slow RWKV + extended Mamba)

**Implementation Required:** Refactor `get_lr_lambda()` in train_v4.py to support per-group lambdas (see guide for pattern)

**Task 38: Balance Regularization Loss**
- **Rationale:** Explicitly penalize R/M ratio deviation from target
- **Implementation:**
  ```python
  # In training loop, after computing component outputs:
  rwkv_var = out_rwkv.var()
  mamba_var = out_mamba.var()
  balance_ratio = rwkv_var / (mamba_var + 1e-8)
  balance_loss = (balance_ratio - 1.0).pow(2)  # Target R/M = 1.0
  total_loss = ce_loss + 0.01 * balance_loss
  ```
- **Expected:** Forces model to maintain balance at cost of some CE loss

**Task 40: BPE Benchmark Validation** ✅ COMPLETE (2026-01-10)

**Command:**
```bash
python train_v4.py --model GF-MH --data data/fineweb_5m.txt --tokenizer bpe --max-steps 5000
```

**Final Results:**
| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| R/M Ratio | 0.21 | 0.20-0.46 | ✅ In range (barely) |
| Train Loss | 4.92 | Decreasing | ✅ Decreasing |
| Val Loss | 6.22 | — | Higher than char-level |
| Train PPL | ~138 | — | — |
| Val PPL | ~505 | — | — |
| Throughput | ~31K tok/s | — | ✅ Stable |
| Steps | 5000/5000 | — | ✅ Complete |

**Critical Observation:** 
- R/M ratio 0.21 is at the **lower bound** of target range (0.20-0.46)
- Activation variance ratio: 71x (RWKV var=8.58, Mamba var=0.12)
- **RWKV still dominates even with BPE tokenization**
- Val loss (6.22) is significantly higher than char-level runs (~1.6)

**Conclusion:** 
⚠️ **BPE did NOT fix component balance as hypothesized.** R/M ratio improved from 0.08-0.11 (char-level) to 0.20-0.21 (BPE), but this is still in WARN territory and activation variance shows severe imbalance (71x).

**Implications:**
1. Char-level vs BPE comparison was valid for showing tokenization matters
2. But BPE alone does NOT solve component balance
3. Fusion variant rankings from Phase 3.6-3.7 need re-validation with BPE
4. Tasks 37-38 (differential warmup, regularization) may still be needed

**Checkpoints saved:**
- `checkpoints/ckpt_GF-MH_step5000.pt`
- `checkpoints/ckpt_GF-MH_final.pt`

**Gate:** ⚠️ Phase 3.8 INCONCLUSIVE — BPE improves balance 2x vs char-level but does not resolve imbalance

---

### Phase 4.0: BPE Re-Validation (NEW — REQUIRED BEFORE SCALING)

**Purpose:** All prior validation (Phases 3.6-3.7) used char-level tokenization. Before proceeding to 30M scaling, we must:
1. **Verify state space fundamentals (S0-S4)** — State machinery works
2. Verify 3.5M Tiny model graduation criteria with BPE
3. Re-run G1-G4 gates with BPE
4. Re-validate fusion variant rankings with BPE
5. Determine if component imbalance is truly architectural or still signal-based

**Rationale:** Char-level was appropriate for quick infrastructure validation, but BPE is the correct baseline for production models. State space monitoring should have been our first priority, not capabilities.

**Key Learning from Task 40:**
- Activation variance ratio: 71x (RWKV var=8.58, Mamba var=0.12)
- This indicates the state machinery may not be functioning as designed
- We need to understand state spaces before testing capabilities

| # | Task | Status | Complexity | Details |
|---|------|--------|------------|---------|
| 41 | Create test_tiny_graduation.py | ⬜ TODO | M | Include S0-S4 state tests + G1-G4 gates |
| 41a | Implement state extraction API | ✅ DONE | M | GF-MH model has return_states=True |
| 42 | Run S0-S4 state space tests | ⬜ TODO | S | Verify state machinery before capabilities |
| 43 | Run Tiny overfit test (BPE) | ⬜ TODO | S | 10-100 samples, loss → near 0 |
| 44 | Run Tiny naive baseline test (BPE) | ⬜ TODO | S | Val loss < random prediction |
| 45 | Run G1-G4 gates (BPE) | ⬜ TODO | M | Re-validate with BPE tokenization |
| 46 | Checkpoint/resume test (BPE) | ⬜ TODO | S | Save + reload works |
| 47 | Fusion variant re-ranking (BPE) | ⬜ TODO | L | 1K steps each: GF, GF-MH, HGF, HY, CP |
| 48 | Component balance assessment | ⬜ TODO | M | Investigate 71x activation ratio |
| 49 | Propagate state API to all models | ⬜ **NEW** | M | 7 model files need return_states |
| 50 | Add state monitoring to train_v4.py | ⬜ **NEW** | M | Integrate return_states in training loop |
| 51 | True Mamba SSM state extraction | ⬜ **NEW** | L | Extract [B, nheads, headdim, d_state] |

**Tiny Model Graduation Criteria (per SCALING_MILESTONES.md):**
| Test | Criteria | BPE Status |
|------|----------|------------|
| **S0-S4 state tests** | State machinery verified | ⏳ **API ready, tests pending** |
| Overfit 10-100 samples | Loss → near 0 | ❓ Not tested |
| Val < naive baseline | Better than random | ❓ Not tested |
| G1-G4 gates pass | Per V4_TESTING.md | ❓ Not tested with BPE |
| Checkpoint/resume works | Save + reload | ❓ Not tested with BPE |
| Gradient flow | All components receiving gradients | ✅ Task 40 shows gradients flowing |
| Component balance documented | Ratio and variance recorded | ⚠️ 71x variance (severe) |

**Order of Operations:**
1. ~~**Task 41a** — Implement state extraction API (BLOCKER)~~ ✅ DONE
2. **Task 42** — Run S0-S4 state space tests
3. **Tasks 43-46** — Run remaining graduation tests
4. **Task 47** — Re-rank fusion variants with BPE
5. **Task 48** — Deep-dive component balance investigation
6. **Task 49** — Propagate state API to all model variants (can parallel with 42-48)
7. **Task 50** — Add state monitoring to training loop
8. **Task 51** — True Mamba SSM state (research, lower priority)

**Gate:** Phase 4.0 PASS when S0-S4 pass AND all Tiny graduation criteria verified with BPE tokenization.

**Next Phase:** If Phase 4.0 PASS → Continue to Phase 3.9 diagnostics with BPE. If FAIL → Debug architecture at 3.5M before scaling.

---

#### Task 41a: State Extraction API

**Status:** ✅ COMPLETE (2026-01-10, Build Session 16)  
**Complexity:** M (Medium)  
**Time:** ~2 hours  

**Purpose:** Enable S0-S4 state tests by exposing internal states through model forward().

**Implementation:**
- RWKV: Modified `_wkv_sequential()` to return final state `[B, H, S]`
- Mamba: Returns output activation proxy `[B, hidden]` (true SSM state deferred)
- Model: Added `return_states=True` parameter to forward()

**API:**
```python
logits, states = model(x, return_states=True)
# states['rwkv_state'].shape = [batch, heads, head_size]
# states['mamba_state'].shape = [batch, hidden_size]  
# states['gate'] = float (avg gate value)
```

**Files Modified:** rwkv6_prototype.py, fla_replacements.py, models/hybrid_v4_ratio.py

**Limitation:** Only GF-MH model modified. Task 49 propagates to all variants.

---

#### Task 49: Propagate State API to All Models

**Status:** ✅ COMPLETE  
**Complexity:** M (Medium)  
**Time:** ~30 minutes (batch edits)  
**Dependencies:** Task 41a ✅  
**Completed:** 2026-01-10

**Purpose:** All model variants need the same `return_states` API for uniform testing.

**Files to Modify:**

| File | Model(s) | Block Class |
|------|----------|-------------|
| `models/hybrid_v4.py` | HY, TINY, SMALL | `ParallelHybridBlock` |
| `models/hybrid_v4_GF.py` | GF | `ParallelHybridBlock_GF` |
| `models/hybrid_v4_WS.py` | WS | `ParallelHybridBlock_WS` |
| `models/hybrid_v4_RF.py` | RF | `ParallelHybridBlock_RF` |
| `models/hybrid_v4_CP.py` | CP | `ParallelHybridBlock_CP` |
| `models/hybrid_v4_HGF.py` | HGF, HGF-MH, HGF-RH | `ParallelHybridBlock_HGF` |
| `models/hybrid_v4_8m.py` | MEDIUM | (check block class) |

**Pattern to Apply:**
```python
# In each block's forward():
def forward(self, x, return_activations=False, return_states=False, ...):
    if return_states:
        out_rwkv, _, rwkv_state_dict = self.rwkv6(norm_x, return_state=True)
        out_mamba, mamba_state_dict = self.mamba2(norm_x, return_state=True)
    else:
        out_rwkv, _, _ = self.rwkv6(norm_x)
        out_mamba = self.mamba2(norm_x)
    ...
    if return_states:
        return x, {'rwkv_state': ..., 'mamba_state': ..., 'gate': ...}

# In each model's forward():
def forward(self, x, return_activations=False, return_states=False, ...):
    if return_states:
        for block in self.blocks:
            h, states = block(h, return_states=True, ...)
            all_states.append(states)
        return logits, all_states[-1]
```

---

#### Task 50: State Monitoring in Training

**Status:** ✅ COMPLETE  
**Complexity:** M (Medium)  
**Time:** ~15 minutes  
**Dependencies:** Task 41a ✅  
**Completed:** 2026-01-10

**Purpose:** Add state monitoring alongside activation monitoring in train_v4.py.

**Current State:** `get_activation_diagnostics()` uses `return_activations=True`

**Changes Needed:**

1. Add `get_state_diagnostics()` function:
```python
@torch.no_grad()
def get_state_diagnostics(model, x):
    """Capture internal states for S0-S4 style monitoring."""
    model.eval()
    logits, states = model(x, return_states=True)
    
    diagnostics = {
        'rwkv_state_norm': states['rwkv_state'].norm().item() if states.get('rwkv_state') is not None else None,
        'mamba_state_norm': states['mamba_state'].norm().item() if states.get('mamba_state') is not None else None,
        'state_ratio': ...,  # rwkv_norm / mamba_norm
        'gate': states.get('gate', 0.5),
    }
    model.train()
    return diagnostics
```

2. Add periodic state logging in training loop (every eval_interval)

3. Add CLI flag `--log-states` to enable state monitoring

**Output Format:**
```
Step 500 | loss=1.59 | R/M=0.23 | rwkv_state_norm=221.3 | mamba_state_norm=4.8 | ratio=46.1x
```

---

#### Task 51: True Mamba SSM State Extraction

**Status:** ⬜ TODO (RESEARCH)  
**Complexity:** L (Large)  
**Time:** ~4-8 hours  
**Dependencies:** Task 41a ✅
**Priority:** Low (output proxy works for diagnostics)

**Purpose:** Extract true Mamba SSM internal state `[B, nheads, headdim, d_state]` instead of output proxy.

**Current Limitation:** Mamba2 forward() only returns states via `inference_params` mechanism designed for generation.

**Research Needed:**
1. Study `mamba_chunk_scan_combined` with `return_final_states=True`
2. Understand `inference_params.key_value_memory_dict` structure
3. Determine if training-time state extraction is possible without perf impact

**Implementation Options:**
- A: Modify `mamba_chunk_scan_combined` call in Mamba2.forward()
- B: Create separate state extraction method (slower, for diagnostics only)
- C: Accept output proxy as "good enough" for validation

**Decision:** Defer until Tasks 42-48 complete. Output proxy may be sufficient.

---

#### Task 25: LRD Test Script

**Status:** ✅ COMPLETE (2026-01-10)  
**Complexity:** S (Small)  
**Time:** ~20 min  

**Created:** `tests/test_lrd.py`

**What it measures:**
- Perplexity at context lengths 8, 16, 32, 64, 128, 256
- "Context benefit" = (loss@8 - loss@128) / loss@8 × 100%
- Higher = model uses long context better

**Key Findings (HY model, step 5000):**
- PPL improves 21-43% from 8→64 context
- Model DOES use long-range context effectively
- This is the right test for char-level models (not NIAH)

**Usage:**
```bash
python tests/test_lrd.py --model HY --checkpoint checkpoints/ckpt_HY_step5000.pt
```

---

#### Task 26: Fusion Variant LRD Comparison

**Status:** ⬜ NEXT  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Priority:** 🔴 BLOCKER - Needed before expensive training

**Goal:** Compare ALL fusion variants on LRD test (both untrained and after 1K steps):

**Variants to test:**
1. HY - Hybrid per-channel gains
2. GF - Gated fusion
3. GF-MH - Gated fusion, Mamba-heavy (Phase 2 winner)
4. HGF - Hybrid-gated fusion (new)
5. CP - Concatenate + Project

**Metrics to collect (per variant):**

| Metric | Untrained | @1K steps | @5K steps |
|--------|-----------|-----------|-----------|
| LRD context benefit (8→128) | % | % | % |
| Val loss | - | value | value |
| Val PPL | - | value | value |
| Gradient ratio | - | value | value |
| Training tok/s | - | value | value |

**Acceptance Criteria:**
- [ ] All 5 variants tested on LRD
- [ ] Results table in V4_FUSION_MODELS.md
- [ ] Top 2 variants identified for extended training

---

#### Task 27: NIAH Test Script

**Status:** ⚠️ DEPRIORITIZED  
**Complexity:** S (Small)  
**Note:** Script exists but test is not appropriate for char-level models

**Problem (discovered Session 13):**
- Char-level models predict next char based on language patterns
- They don't "retrieve" injected patterns (0% accuracy expected)
- After "X" in "XYZQWK", model predicts "a" (38.7%) because "Xa" is common

**Future Option (word-level):**
- Requires BPE tokenizer
- Requires instruction-following training data
- Add to Phase 5 or later

---

#### Task 28: Quick Train Comparison

**Status:** ⬜ PENDING  
**Complexity:** M (Medium)  
**Time:** ~45 min (1K steps × 5 variants)  
**Depends On:** Task 26 (know which variants to focus on)

**Goal:** Run 1K training steps on each fusion variant, measure:
- Final loss/PPL
- Training speed (tok/s)
- Gradient ratio
- Memory usage

**Config:**
```yaml
model: [variant]
batch_size: 32
seq_len: 64
max_steps: 1000
eval_every: 100
```

---

#### Task 29: Component Balance Tests

**Status:** ⬜ PENDING  
**Complexity:** M (Medium)  
**Time:** ~30 min  
**Depends On:** Task 28

**Goal:** After 1K training, measure component balance for each variant:

| Metric | Target Range | How to Measure |
|--------|--------------|----------------|
| Gradient ratio (Mamba/RWKV) | 0.3-3.0 | train_v4.py logs |
| Activation variance ratio | <10x | Custom script |
| Gate distribution | 0.3-0.7 mean | For GF/HGF variants |

**Acceptance Criteria:**
- [ ] All variants' balance metrics documented
- [ ] Identify variants with poor balance
- [ ] Recommend tuning for imbalanced variants

---

#### Task 30: Fusion Variant Ranking

**Status:** ⬜ PENDING  
**Complexity:** S (Small)  
**Time:** ~15 min  
**Depends On:** Tasks 26, 28, 29

**Goal:** Create final ranking table combining all metrics:

| Rank | Variant | LRD Score | Val Loss | Grad Ratio | Recommendation |
|------|---------|-----------|----------|------------|----------------|
| 1 | ? | ? | ? | ? | Extended training |
| 2 | ? | ? | ? | ? | Extended training |
| 3 | ? | ? | ? | ? | Monitor |
| 4-5 | ? | ? | ? | ? | Deprioritize |

**Output:** Update V4_FUSION_MODELS.md with ranking and recommendation.

---

### Phase 3.9: Validation-First Approach (Prerequisite to Phase 4)

**Goal:** Master 8M model completely before scaling to 30M. Deploy diagnostic tools to understand stateful component behavior.

**Principle:** Every architectural weakness magnifies with scale. Fix at 8M (cheap) rather than 30M (expensive).

**Timeline:** 3 weeks (Jan 10-31)  
**Status:** ⬜ PLANNING (detailed roadmap in [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md))

#### Tasks (Weekly Breakdown)

| Week | Task | Status | Complexity | Deliverable |
|------|------|--------|------------|------------|
| 1 | Deploy State Tracing Module | ⬜ | M | `tools/state_tracing.py` |
| 1 | Run 4 Diagnostic Tests | ⬜ | M | `eval/diagnostic_tests.py` + results |
| 1 | Create Statefulness Report | ⬜ | S | `reports/statefulness_report_week1.md` |
| 2 | Build 3 Validation Tools | ⬜ | L | Health Monitor, Coupling Analyzer, Info Flow Tracer |
| 2 | Establish Baselines | ⬜ | M | `metrics/baseline_8m_metrics.json` |
| 2 | Define Thresholds | ⬜ | S | "Good enough" metrics for 8M graduation |
| 3 | Fix Issues (if any) | ⬜ | L/M | Architectural adjustments based on Week 1-2 findings |
| 3 | Re-validate (if fixes) | ⬜ | M | Confirm fixes work, metrics improve |
| 3 | Go/No-Go Decision | ⬜ | S | `VALIDATION_GATE_PASS.md` or `VALIDATION_GATE_FAIL.md` |

#### Success Criteria for 30M Scaling

✅ **All Go/No-Go conditions met:**
- State health metrics pass (no divergence/collapse)
- Component coupling balanced (ratio 0.2-5.0)
- Information flow shows both components active (>0.2 each)
- All 4 diagnostic tests pass
- Extended training (10K steps) remains stable
- BPE validation (Task 40) shows expected results

❌ **No-Go triggers:**
- State divergence detected
- One component dead (coupling <0.1)
- Diagnostic test failure
- Unexpected behavior under stress

#### Why This Phase Matters

Three weeks of validation now prevent weeks of debugging at 30M scale.

#### Related Documents

- **Strategic Framework:** [V4.5_VALIDATION.md](V4.5_VALIDATION.md) — Mindset shift, open-source opportunities
- **Execution Plan:** [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md) — Detailed 3-week breakdown
- **Implementation Guide:** [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md) — Concrete test code, state tracing module, grounding calculator, diagnostic tests

---

### Phase 4: Conditional Scaling & Extended Evaluation (After Phase 3.9 Gate)

**Prerequisite:** Must pass Phase 3.9 validation gate.

**Overview:**
- If Phase 3.9 PASS: Proceed with 30M model design
- If Phase 3.9 FAIL: Return to Phase 3 architecture redesign

**Conditional Tasks:**

| Task | Depends On | Status | Complexity | Details |
|------|------------|--------|------------|---------|
| 30M Model Design | Phase 3.9 PASS | ⬜ | M | Scale architecture with lessons learned from 8M |
| 30M Training (50K steps) | Task 30M Design | ⬜ | XL | Extended training with validation tools active |
| NIAH Evaluation (16K tokens) | Task 30M Training | ⬜ | L | Long-context retrieval test |
| LongBench Evaluation | NIAH PASS | ⬜ | L | Multitask real-world memory (50K-100K tokens) |
| InfiniteBench Evaluation | LongBench PASS | ⬜ | XL | Ultra-long context (100K-200K tokens) |

**Gate:** Phase 4 complete when 30M model trained and shows expected scaling behavior vs 8M baseline.

---

## Milestone Goals: Scale-Specific Objectives

**Strategic Framework:** Each scale target has specific architectural validation, conversational goals, and context limits. Success criteria must be met before scaling.

| Scale | Architecture Test | Conversational Goal | Context Target | Success Metric |
|-------|------------------|-------------------|-----------------|----------------|
| **3.5M** | Fusion stable training | Basic Q&A coherence | 512 tokens | C1a: >95% pass (1-turn recall) |
| **8M** | State persistence >3 turns | Role anchoring | 1024 tokens | C3a+C3b: >85% pass (state tracking) |
| **30M** | Multi-document grounding | Persona consistency | 2048 tokens | C2a+C5b: >80% pass (grounding+persona) |
| **125M** | Infinite-context patterns | Expert-level dialogue | 4096+ tokens | C6a+C2b: >75% pass (isolation+synthesis) |

**Test Suite Reference:** See [CANARY_TESTS.md](CANARY_TESTS.md) for detailed implementations. [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md) specifies which suite (minimal/standard/extended) runs each week.

---

## Evaluation Infrastructure: Critical Tools to Build

**Rationale:** Most teams under-invest in evaluation. For state-based architecture, evaluation is harder than training.

### Early Builds (Phase 3.9)

| Tool | Purpose | Week | Priority | Impact |
|------|---------|------|----------|--------|
| State Visualization | Visualize hidden states across conversations | 1 | 🔴 CRITICAL | Reveals state dynamics (e.g., divergence, collapse) |
| Context Heat Map | Show which context parts are "attended" | 2 | 🟡 HIGH | Identifies attention patterns in state updates |
| Drift Detector | Automatically flag persona inconsistencies | 2 | 🟡 HIGH | Early warning of role degradation |
| Canary Test Harness | Run C1-C6 tests, collect JSON results | 1 | 🔴 CRITICAL | Objective scaling decisions |

### Later Builds (Phase 4)

| Tool | Purpose | Phase | Priority | Impact |
|------|---------|-------|----------|--------|
| Long-Context Benchmark Suite | Custom dataset of progressively harder grounding tasks | 4 | 🟡 HIGH | Validates 2K-10K token reasoning |
| Episodic Memory Tracer | Track how conversational facts are encoded in state | 4 | 🟡 HIGH | Unique advantage for hybrid architecture |
| Checkpoint Comparator | Contrast checkpoint expansion vs fresh training | 4 | 🟡 HIGH | Validates Phase 4 scaling strategy |

**Build Order:** State Visualization → Canary Test Harness → Drift Detector → Context Heat Map

---

## When to Graduate: Decision Matrix

**For each scale-up decision, score on 0-3 scale:**

| Category | 0 (Stop) | 1 (Risk) | 2 (OK) | 3 (Excellent) | Weight |
|----------|----------|----------|--------|---------------|---------
| **Architecture Stability** | NaN/Inf losses | Spiky training | Smooth curves | Very stable | x1.5 |
| **Canary Performance** | <50% pass | 50-75% pass | 75-85% pass | >85% pass | x1.5 |
| **Scaling Law Fit** | No pattern | Noisy | Clear trend | Linear/predictable | x1.0 |
| **Resource Prediction Confidence** | "No idea" | ±50% | ±20% | ±10% | x1.0 |

**Decision Rule:**
- **GO (Graduate):** Total ≥10 AND no category is 0
- **CAUTION:** Total 8-9 (review highest-risk category)
- **HOLD (Don't Scale):** Any category is 0 OR total <8

**Example:**
```
3.5M → 8M decision:
- Architecture Stability: 3 (very smooth training)
- Canary Performance: 3 (C1a-C1b both >90%)
- Scaling Law Fit: 2 (clear loss trend, some noise)
- Resource Prediction: 2 (±20% VRAM estimate)

Total = 3×1.5 + 3×1.5 + 2×1.0 + 2×1.0 = 13 ✅ GO
```

---

### Phase 5: Advanced Long-Context Evaluation (Optional - "Later Game")

**Prerequisites:** Must pass Phase 4 evaluation (NIAH >80% accuracy at 16K tokens) before attempting these benchmarks.

| # | Task | Status | Depends On | Gates | Complexity | Source/Details |
|---|------|--------|------------|-------|------------|----------------|
| 25 | LongBench Evaluation | ⬜ PENDING | Phase 4 PASS | - | L | [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Multitask 50K-100K |
| 26 | Analyze LongBench Results | ⬜ PENDING | Task 25 | F1 >0.6 @ 100K | M | Needs detailed description |
| 27 | InfiniteBench Evaluation | ⬜ PENDING | Task 26 (LongBench pass) | - | XL | [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Ultra-long 100K-200K |

| 28 | Long-Context Optimization Pass | ⬜ PENDING | Task 27 | - | XL | Memory tuning, 1-2 weeks |
| 29 | Document Long-Context Capabilities | ⬜ PENDING | Task 28 | - | M | Needs detailed description |

**Gate:** Phase 5 complete when model demonstrates production-ready long-context memory (F1 >0.6 at 100K tokens).

**Warning:** Phase 5 requires significant compute resources (80GB+ VRAM for 1M token contexts). Consider gradient checkpointing, quantization, or cloud resources.

**Legend:** ⬜ PENDING | 🔄 IN HANDOFF | ✅ COMPLETE

---

## Completed Tasks (Reference Only)

<details>
<summary>Tasks 1-5: Foundation Setup (Click to expand)</summary>

### Task 1: Verify data_v030.py + tokenizer_v030.py Work
**Status:** ✅ COMPLETE (2026-01-09)  
Files now in `archive/` but still functional. train_v4.py imports from archive.

### Task 2: Create/Validate Test Dataset
**Status:** ✅ COMPLETE (2026-01-09)  
Dataset: shakespeare.txt (1.1M tokens, char-level, 97 vocab size)

### Task 3: Build First Hybrid (ParallelHybridBlock)
**Status:** ✅ COMPLETE (2026-01-09)  
Output: `hybrid_v4.py` - 3.8M parameters, RWKV6 + Mamba2 parallel architecture

### Task 4: Define Training Configuration
**Status:** ✅ COMPLETE (2026-01-09)  
Output: `train_v4.py` - Full training script with gradient monitoring, differential LR

### Task 5: First Training Run
**Status:** ✅ COMPLETE (2026-01-09)  
Results: Loss 1.37, perplexity 3.0, 33K tok/s, gradient ratio warning (0.15-0.16)

</details>

---

## Archived Task Details (Reference Only)

> **Note:** Tasks 6.5-6.12 were SUPERSEDED by Phase 0 CUDA integration. 
> These descriptions are kept for historical reference only.
> See Phase 0 and Phase 1 tables above for actual completion status.

### Task 6.6: Research RWKV-6 Architecture

**Status:** ✅ SUPERSEDED by Task 0.1  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Scope:** Find authoritative RWKV-6 specifications and document requirements

**Decision Context:**
- FLA library not installed, custom wrappers in `archive/fla_replacements.py` are simplified/fake
- Cannot use FLA (per user decision)
- Must build correct RWKV-6 and Mamba-2 from scratch
- **Critical:** V3 failed by making up components - we must get this right

**Research Objectives:**
1. Find official RWKV-6 paper/specification
2. Identify key architectural components:
   - Time mixing mechanism (WKV kernel)
   - Channel mixing
   - Token/time shift
   - State management
3. Document mathematical formulas
4. Find reference implementations (RWKV-LM official repo)
5. Identify critical differences vs RWKV-5

**Deliverables:**
- Create `RWKV6_SPEC.md` with:
  - Paper citations
  - Architecture diagram/description
  - Mathematical formulas
  - Key parameters (num_heads, head_dim, etc.)
  - Critical implementation notes
  - Links to reference code

**Acceptance Criteria:**
- [ ] Official RWKV-6 paper/spec found and documented
- [ ] Key components identified and described
- [ ] Mathematical formulas transcribed
- [ ] Reference implementation links saved
- [ ] RWKV6_SPEC.md created in workspace

**Resources:**
- RWKV official GitHub: https://github.com/BlinkDL/RWKV-LM
- Papers: arXiv search for "RWKV-6"
- Community: RWKV Discord/discussions

---

### Task 6.7: Research Mamba-2 Architecture

**Status:** ✅ SUPERSEDED by Task 0.2  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Scope:** Find authoritative Mamba-2 specifications and document requirements

**Research Objectives:**
1. Find official Mamba-2 paper (Dao & Gu)
2. Identify key components:
   - State-space model (SSM) formulation
   - Selective scan mechanism
   - SSD (Structured State-Space Duality)
   - Chunk-wise processing
3. Document mathematical formulas
4. Find reference implementations (state-spaces/mamba repo)
5. Hardware requirements (CUDA kernels, Triton)

**Deliverables:**
- Create `MAMBA2_SPEC.md` with:
  - Paper citations
  - Architecture description
  - SSM formulation
  - Key parameters (d_state, d_conv, expand)
  - Implementation notes
  - Links to reference code

**Acceptance Criteria:**
- [ ] Official Mamba-2 paper found and documented
- [ ] SSM mechanism understood and described
- [ ] Mathematical formulas transcribed
- [ ] Reference implementation links saved
- [ ] MAMBA2_SPEC.md created in workspace

---

### Task 6.8: Audit Custom Wrappers

**Status:** ✅ SUPERSEDED by Phase 0  
**Complexity:** M (Medium)  
**Time:** ~1-2 hours  
**Scope:** Compare fla_replacements.py against official specs

**What to Check:**

**RWKV6Attention (lines 9-62 in fla_replacements.py):**
- [ ] Time mixing uses correct WKV formula
- [ ] Channel mixing implemented correctly
- [ ] Token shift mechanism present
- [ ] State management correct
- [ ] Return signature matches: (output, attn_weights, past_kv)

**Mamba2 (lines 65-113 in fla_replacements.py):**
- [ ] SSM formulation correct
- [ ] Selective scan mechanism present
- [ ] Conv1d usage correct (d_conv parameter)
- [ ] Expansion factor correct
- [ ] State management correct

**Deliverables:**
- Create `WRAPPER_AUDIT.md` documenting:
  - What's correct
  - What's missing
  - What's wrong
  - Severity of issues (blocker/warning/minor)
  - Recommendations (fix vs rebuild)

**Acceptance Criteria:**
- [ ] Line-by-line comparison completed
- [ ] All discrepancies documented
- [ ] Severity assessed for each issue
- [ ] Clear recommendation: fix wrappers or rebuild from scratch

---

### Task 6.9: Implement/Fix RWKV-6 Component

**Status:** ✅ SUPERSEDED by Task 0.1  
**Complexity:** L/XL (Large or Extra Large - TBD after audit)  
**Time:** ~2-8 hours (depends on audit results)  
**Scope:** Build correct RWKV-6 implementation

**Will be detailed after Task 6.8 audit determines scope.**

**Acceptance Criteria:**
- [ ] Passes G1 gate (forward pass, no NaN)
- [ ] Matches RWKV-6 spec exactly
- [ ] Documented in V4_BUILD_LOG.md

---

### Task 6.10: Implement/Fix Mamba-2 Component

**Status:** ✅ SUPERSEDED by Task 0.2  
**Complexity:** L/XL (Large or Extra Large - TBD after audit)  
**Time:** ~2-8 hours (depends on audit results)  
**Scope:** Build correct Mamba-2 implementation

**Will be detailed after Task 6.8 audit determines scope.**

**Acceptance Criteria:**
- [ ] Passes G1 gate (forward pass, no NaN)
- [ ] Matches Mamba-2 spec exactly
- [ ] Documented in V4_BUILD_LOG.md

---

### Task 6.11: Rebuild hybrid_v4.py

**Status:** ✅ COMPLETE (fla_replacements.py)  
**Complexity:** L (Large)  
**Time:** ~2-3 hours  
**Scope:** Integrate verified RWKV-6 and Mamba-2 components

**What to Do:**
1. Move verified wrappers to root as `rwkv6_component.py` and `mamba2_component.py`
2. Update hybrid_v4.py imports
3. Verify ParallelHybridBlock architecture unchanged
4. Test forward pass
5. Run G1 and G2 gates

**Acceptance Criteria:**
- [ ] Imports work: `from rwkv6_component import RWKV6Attention`
- [ ] Passes G1 gate (forward pass)
- [ ] Passes G2 gate (init entropy 2.0-5.0)
- [ ] Model parameters ~3.8M (same as before)
- [ ] Documented in V4_BUILD_LOG.md

---

### Task 6.12: Verify Model Works

**Status:** ✅ COMPLETE (test_phase0_complete.py)  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Scope:** Full validation before proceeding to optimization

**Tests:**
1. Import test: `import hybrid_v4` succeeds
2. Instantiation: Model builds without errors
3. Forward pass: Process sample batch
4. G1 gate: No NaN, correct output shapes
5. G2 gate: Init entropy in healthy range
6. Quick training: 10 steps to verify gradients flow

**Acceptance Criteria:**
- [ ] All tests pass
- [ ] Gates G1-G2 passed
- [ ] Ready for Task 6.5 (monitoring test)
- [ ] Results documented in V4_BUILD_LOG.md

---

### Task 6.5: Test Monitoring Tools

**Status:** ✅ COMPLETE  
**Complexity:** S (Small)  
**Time:** ~15-20 minutes  
**Scope:** Verify monitoring tools work during actual training

**Why This Task:**
Standard operating procedure - we installed tools but didn't verify they work together under load. Need to test before using them for profiling analysis.

**What to Do:**
1. Start nvidia-smi logging in background: `nvidia-smi -l 1 > logs/monitoring/test_run_$(date +%Y%m%d_%H%M%S).log &`
2. Run short training session: `python train_v4.py --steps 100 --no-checkpoint`
3. Monitor in real-time with: `nvtop` (in separate terminal)
4. Stop nvidia-smi logger after training completes
5. Verify log file captured GPU metrics during training

**Acceptance Criteria:**
- [ ] Training runs successfully with monitoring active
- [ ] nvidia-smi log file shows GPU utilization >0% during training
- [ ] nvtop displays real-time stats (can screenshot or describe observed metrics)
- [ ] Baseline under load documented: GPU util %, VRAM usage, temperature, power draw
- [ ] No conflicts or performance degradation from monitoring tools

**What to Document in V4_BUILD_LOG.md:**
- GPU utilization % during training (target: should see >50%)
- VRAM usage under load (expect ~422 MiB based on previous run)
- Temperature and power draw peaks
- Any anomalies or issues with monitoring tools

---

### Task 7: Baseline Performance Profiling

**Status:** ⬜ **NEXT**  
**Complexity:** M (Medium)  
**Time:** ~1 hour  
**Scope:** Create reusable benchmark suite + record baseline measurements

**Why This Matters:**
- Cannot optimize without knowing current performance
- Need fixed benchmark configs for reproducible comparisons
- Existing metrics scattered across train_v4.py + docs, need consolidation

**Existing Assets:**
- `train_v4.py`: Runtime metrics (tok/s, loss, R/M ratio, entropy, activation stats)
- `V4_DIAGNOSTICS.md`: KernelBenchmark class specification (lines 545-700)
- `V4_TESTING.md`: Benchmark snippets (speed_test, memory_test)
- `test_phase0_complete.py`: Reusable test pattern (7 tests + 5 gates)

**Sub-Tasks:**

| # | Sub-Task | Status | Output | Complexity |
|---|----------|--------|--------|------------|
| 7.1 | Create benchmark_suite.py | ⬜ | Reusable benchmark script | M |
| 7.2 | Run B1: Throughput test | ⬜ | tok/s at fixed batch/seq | S |
| 7.3 | Run B2: Memory test | ⬜ | Peak VRAM at fixed config | S |
| 7.4 | Run B3: Stability test | ⬜ | 100-step loss delta | S |
| 7.5 | Document baseline in V4_BUILD_LOG.md | ⬜ | Session 9 entry | S |

**Benchmark Definitions (Fixed Configs):**

| Benchmark | Config | Metric | Target | Pass Criteria |
|-----------|--------|--------|--------|---------------|
| B1: Throughput | batch=8, seq=64, 100 steps | tok/s | Measure baseline | Record value |
| B2: Memory | batch=8, seq=64 | Peak VRAM (MiB) | <1000 MiB | Record value |
| B3: Stability | batch=8, seq=64, 100 steps | Loss Δ | Decreasing | loss_end < loss_start |

**Acceptance Criteria:**
- [ ] benchmark_suite.py created (reusable like test_phase0_complete.py)
- [ ] All 3 benchmarks run with fixed configs
- [ ] Baseline numbers recorded in V4_BUILD_LOG.md Session 9
- [ ] Ready for Task 8 (optimizations) comparison

---

### Task 8: Apply Quick Win Optimizations

**Status:** ⬜ PENDING  
**Complexity:** L (Large)  
**Time:** ~2-3 hours  
**Scope:** Test each optimization using benchmark_suite.py from Task 7

**Dependencies:** Task 7 (baseline measurements required for comparison)

**Optimizations to test (one at a time):**

| # | Optimization | Config Change | Expected Improvement |
|---|--------------|---------------|----------------------|
| 8.1 | Larger batch | batch: 8 → 16 | ~1.5-2x tok/s |
| 8.2 | DataLoader workers | workers: 0 → 4, pin_memory=True | ~1.2x tok/s |
| 8.3 | Mixed precision (AMP) | torch.cuda.amp | ~1.5-2x tok/s |
| 8.4 | torch.compile | model = torch.compile(model) | ~1.3-2x tok/s |

**Testing Protocol:**
1. Run benchmark_suite.py with ONE change
2. Compare to Task 7 baseline (B1, B2, B3)
3. If B3 passes (loss decreasing), optimization is valid
4. Record improvement factor

**Acceptance Criteria:**
- [ ] Each optimization tested independently with benchmark_suite.py
- [ ] Comparison table with baseline vs each optimization
- [ ] No quality degradation (B3 must still pass)
- [ ] Best single optimization identified

---

### Task 10: Run Controlled Experiments

**Status:** ⬜ PENDING  
**Time:** ~3-4 hours  
**Scope:** Systematic testing of optimization combinations (V4.5_OPTIMIZATION.md Phase 4)

**Experiment matrix:**
- Baseline (8 batch, no workers, fp32, no compile)
- +Batch size increase
- +Workers
- +AMP
- +torch.compile (all optimizations)

**Acceptance Criteria:**
- [ ] Run each config for 1000 steps
- [ ] Log comparison table with throughput, VRAM, loss
- [ ] Validate no quality degradation (loss within ±0.05)
- [ ] Document speedup multiplier

---

### Task 11: Select Optimal Configuration

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Choose best setup and update train_v4.py defaults

**Requirements:**
- Compare all experiment results
- Choose configuration with best throughput/quality trade-off
- Update train_v4.py CONFIG with optimal settings
- Document final configuration

**Acceptance Criteria:**
- [ ] Optimal config selected (target: 5x baseline throughput)
- [ ] train_v4.py updated with new defaults
- [ ] Results documented in V4_BUILD_LOG.md

---

### Task 11: Analyze Training Results

**Status:** ⬜ PENDING  
**Time:** ~1-2 hours  
**Scope:** Deep analysis of baseline and optimized training runs with proper monitoring tools

**Requirements:**
- Review baseline training curves (5000 steps)
- Review optimized training curves (from Task 9-10)
- Compare gradient dynamics across configurations
- Analyze component contributions (RWKV vs Mamba)
- Check for activation collapse or state issues

**What We Now Have for Analysis:**
- Performance profiles (torch.profiler traces)
- GPU utilization data (nvtop logs)
- Throughput comparisons (baseline vs optimized)
- Loss convergence curves

**Acceptance Criteria:**
- [ ] Training curves analyzed and documented
- [ ] Gradient ratio patterns understood
- [ ] Component health validated
- [ ] Recommendations for hyperparameter tuning (Task 12)
- [ ] Document findings in V4_BUILD_LOG.md

---

### Task 12: Address Gradient Imbalance

**Status:** ⬜ PENDING  
**Time:** ~1-2 hours  
**Scope:** Fix RWKV/Mamba gradient ratio (currently 0.15, target 0.3-3.0)

**Analysis needed:**
- Why is RWKV gradient 6-7x weaker than Mamba?
- Is mamba_lr_mult=2.0 too high?
- Does RWKV need better initialization?

**Potential fixes:**
- Adjust mamba_lr_mult (try 1.5, 1.0, 0.5)
- Increase RWKV learning rate independently
- Check RWKV6 initialization in hybrid_v4.py

**Acceptance Criteria:**
- [ ] Root cause identified
- [ ] Solution tested (1000 steps)
- [ ] Gradient ratio in healthy range (0.3-3.0)
- [ ] Document fix in V4_BUILD_LOG.md

---

### Task 13: Extended Training Run

**Status:** ✅ COMPLETE (2026-01-09)  
**Time:** 582.4s (~10 min) at 35K tok/s avg  
**Scope:** Train optimized model to convergence (5000 steps)

**Results:**
- Final Train Loss: 1.1375
- Final Val Loss: 1.4916
- Best Val Loss: 1.4607 (step ~4500)
- Final PPL: 3.12 train / 4.44 val
- Entropy: 3.83 → 3.91 (healthy growth)
- Gradient Ratio: Started 0.4-0.5, drifted to 0.29 at end (low LR)
- Checkpoints: 6 saved (1K, 2K, 3K, 4K, 5K, final)

**Acceptance Criteria:**
- [x] Training completes without crashes
- [x] Val loss converges or plateaus (1.46 best)
- [x] Final model checkpoint saved (ckpt_HY_final.pt)
- [x] Training curves logged and analyzed
- [x] Results documented

**Observation:** Gradient ratio drifted from 0.4-0.5 (mid-training) to 0.28-0.33 (late training) as LR decayed via cosine schedule. This is expected behavior - RWKV layers have proportionally lower gradients when LR is very small. Model convergence was excellent regardless.

---

<!-- Legacy task descriptions removed (2026-01-10) - see Phase 1 table for completion status -->

---

### Task 22: Long-Context Retrieval Test (NIAH)

**Status:** ⬜ PENDING  
**Time:** ~1-2 hours  
**Scope:** Validate model's long-context memory retention capability

**What is NIAH?**
Needle-in-a-Haystack test hides key facts ("needles") in filler text ("haystack") and queries retrieval accuracy. Reveals "context rot" where models fail beyond claimed context length.

**Why test this?**
- Hybrid RWKV/Mamba architecture claims better long-context handling than pure attention
- Need empirical validation of context window effectiveness (not just theoretical claims)
- Identifies degradation points before scaling to larger models

**Setup:**
```bash
# Install testing framework
pip install needlehaystack

# Test at multiple context lengths
needlehaystack.run_test \
  --provider custom \
  --model_name "hybrid_v4_8M" \
  --context_lengths "[1000,2000,4000,8000,16000]" \
  --document_depth_percents "[10,25,50,75,90]" \
  --multi_needle False \
  --save_results True
```

**Test protocol:**
1. **Single needle (baseline):** One fact hidden in Paul Graham essays
   - Needle: "The best pizza topping is mushrooms."
   - Query: "What is the best pizza topping?"
   - Test at 1K, 2K, 4K, 8K, 16K tokens

2. **Depth variation:** Place needle at start (10%), middle (50%), end (90%)
   - Check if middle degrades (common attention bias)

3. **Multi-needle (advanced):** 10 facts spaced evenly
   - Tests multi-fact recall and memory interference

**Custom model integration:**
```python
from needlehaystack import LLMNeedleHaystackTester

class HybridModelTester(LLMNeedleHaystackTester):
    def __init__(self, model_path):
        self.model = load_hybrid_model(model_path)
    
    def evaluate_model(self, context: str, question: str) -> str:
        prompt = context + "\n\n" + question
        output = self.model.generate(prompt, max_tokens=50)
        return output
```

**Expected Results:**
- **Target:** >80% accuracy up to claimed context length
- **Baseline comparison:** Document vs GPT-3.5 (degrades at 4K-8K typically)
- **Hybrid advantage:** Should outperform pure attention at 8K+

**Analysis:**
- Plot accuracy heatmap (context length vs needle depth)
- Identify degradation point (where accuracy drops below 80%)
- Compare RWKV-heavy vs Mamba-heavy configurations
- Document in V4_BUILD_LOG.md

**Acceptance Criteria:**
- [ ] Tests run successfully at 1K, 2K, 4K, 8K, 16K tokens
- [ ] Accuracy measured for start/middle/end positions
- [ ] Results visualized (heatmap or line plot)
- [ ] Degradation point identified
- [ ] Results compared to baseline (GPT-3.5 or similar)
- [ ] Findings documented in V4_BUILD_LOG.md

**Red Flags:**
- Accuracy <80% at 4K tokens (indicates poor long-context handling)
- Middle positions significantly worse than start/end
- Multi-needle accuracy <50% (memory interference)

**See Also:** [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Full NIAH methodology and setup

---

### Task 25: LongBench Evaluation (Multitask Real-World Memory)

**Status:** ⬜ PENDING  
**Time:** ~2-4 hours  
**Scope:** Evaluate holistic memory on diverse real-world tasks (QA, summarization, code)

**Prerequisites:**
- Task 22 (NIAH) must pass with >80% accuracy at 16K tokens
- Model demonstrates basic long-context capability

**What is LongBench?**
- Bilingual (English/Chinese) benchmark with 20+ tasks
- Multi-document QA, long code completion, summarization
- Tests up to 200K tokens
- Focus on deep understanding and cross-document reasoning

**Setup:**
```bash
# Clone repository
git clone https://github.com/THUDM/LongBench
cd LongBench

# Install dependencies
pip install -r requirements.txt

# Run evaluation (auto-downloads datasets)
python eval.py --model hybrid_v4_8M --length 100000
```

**Test protocol:**
1. **Start at 50K tokens:** Baseline evaluation on all tasks
2. **Scale to 100K tokens:** Test sustained performance
3. **Select key tasks:** Focus on multidoc_qa, code_completion, summarization
4. **Compare against baselines:** GPT-4 (~0.75 F1), Claude (~0.72 F1)

**Metrics:**
- **F1 score** for QA tasks
- **ROUGE** for summarization
- **Code accuracy** for completion
- Cross-document reasoning capability

**Expected Results:**
- **Target:** F1 >0.6 at 100K tokens
- **Comparison:** Document vs SOTA (GPT-4, Claude, Gemini)
- **Component analysis:** RWKV vs Mamba contribution to long-context

**Acceptance Criteria:**
- [ ] Evaluation runs at 50K and 100K tokens
- [ ] F1/ROUGE scores collected for all tasks
- [ ] Degradation curve plotted (accuracy vs length)
- [ ] Comparison to baseline models documented
- [ ] Task-specific strengths/weaknesses identified
- [ ] Results documented in V4_BUILD_LOG.md

**Red Flags:**
- F1 <0.4 at 50K tokens (model struggles with moderate length)
- Sharp performance cliff beyond certain length
- Specific task failures (e.g., code but not QA)

---

### Task 27: InfiniteBench Evaluation (Ultra-Long Context)

**Status:** ⬜ PENDING  
**Time:** ~4-8 hours  
**Scope:** Test "infinite-like" memory at 100K-200K tokens

**Prerequisites:**
- Task 26 (LongBench analysis) must show F1 >0.6 at 100K tokens
- Adequate hardware (80GB+ VRAM or gradient checkpointing setup)

**What is InfiniteBench?**
- Ultra-long context benchmark (100K-200K+ tokens)
- Complex tasks: book QA, infinite math sequences, full novel summarization
- Probes sustained attention and hierarchical memory
- Tests if model maintains state at extreme lengths

**Setup:**
```bash
# Follow arXiv instructions (https://arxiv.org/abs/2402.13718)
# Generate test data
python generate_infinitebench_data.py --task book_qa --length 150000

# Run evaluation
python eval_infinitebench.py --model hybrid_v4_8M --task book_qa
```

**Test protocol:**
1. **Book QA:** Answer questions about full novels (150K+ tokens)
2. **Math sequences:** Maintain state through infinite series
3. **Code repository:** Navigate entire codebases
4. **Multi-session dialogue:** Track hundreds of conversation turns

**Resource requirements:**
- **VRAM:** 80GB+ recommended (or use gradient checkpointing)
- **Time:** Hours per evaluation task
- **Consider:** Model quantization or memory-efficient attention

**Expected challenges:**
- Extreme VRAM requirements at 200K+ tokens
- Long evaluation time (hours per task)
- May require distributed inference or CPU offloading

**Acceptance Criteria:**
- [ ] Evaluation completes at 100K-150K tokens
- [ ] Accuracy measured for book QA and math sequences
- [ ] Memory usage profiled and documented
- [ ] Graceful degradation observed (not catastrophic failure)
- [ ] RWKV/Mamba component contributions analyzed
- [ ] Results documented in V4_BUILD_LOG.md

**Red Flags:**
- OOM errors below 100K tokens (inadequate memory management)
- Catastrophic accuracy drop (not graceful degradation)
- Specific task complete failures

**See Also:** [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Advanced long-context benchmarks section

---

### Task 28: Long-Context Optimization Pass

**Status:** ⬜ PENDING  
**Time:** ~1-2 weeks  
**Scope:** Systematic optimization based on LongBench/InfiniteBench findings

**What to optimize:**
1. **Memory efficiency:** Gradient checkpointing, quantization, efficient attention
2. **Component balance:** Tune RWKV/Mamba ratio for long-context tasks
3. **State management:** Optimize recurrent state handling at extreme lengths
4. **Training data:** Curate long-context training examples

**Approach:**
- Controlled experiments (one variable at a time)
- Document each optimization's impact
- Maintain validation loss within ±5% of baseline
- Focus on 80K-100K token range (practical use case)

**Deliverables:**
- [ ] Memory-optimized model configuration
- [ ] Long-context training data pipeline
- [ ] Performance vs memory tradeoff analysis
- [ ] Production deployment guidelines
- [ ] Updated V4_BUILD_LOG.md with optimization results

**See Also:** [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Full optimization methodology

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
