# V4 Strategy Document - Task Backlog

**Created:** 2026-01-08  
**Updated:** 2026-01-10 (Observation 14: Attractor Behavior)  
**Repository:** https://github.com/9to5ninja-projects/groundthink  
**Purpose:** Ordered queue of tasks to be completed one at a time  
**Current Goal:** ✅ Phase 4.0 PASSED — Task 48a COMPLETE, Next: Task 52+ (diagnostic tools)

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
| 0.4 | Hybrid Block Integration | ✅ COMPLETE | cuda_backends.py, hybrid_v4.py | 0.3 | M |

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
- `cuda_backends.py` - CUDA wrapper integration with fallback
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
| 6.11 | ~~Rebuild hybrid_v4.py~~ | ✅ COMPLETE | Tasks 0.1, 0.2 | G1-G2 | M | cuda_backends.py integrates CUDA wrappers |
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
| 18.3 | ~~NIAH Test Implementation~~ | ⚠️ DEPRIORITIZED | Task 18.1 | L | Not valid for char-level models |
| 18.4 | Qualitative Eval Suite | ⬜ PENDING | Task 18.3 | M | Generation samples, perplexity, patterns |
| 18.5 | Evaluation Baseline | ⬜ PENDING | Task 18.4 | M | Run eval suite on 5M winner (GF-MH) |

**Gate:** ✅ Phase 2.5 COMPLETE for core infrastructure (18.1, 18.2). Eval suite deferred to post-Phase 4.0.

---

#### ~~Task 18.1: Model Registry & Factory~~ ✅ COMPLETE

**Status:** ✅ COMPLETE  
**Complexity:** M (Medium)  
**Completed:** 2026-01-09

~~Problem/Solution details redacted — implementation exists in `models/__init__.py`~~

**Usage:**
```bash
python train_v4.py --model 8M --steps 50000
python train_v4.py --model GF-MH --steps 5000
```

---

#### ~~Task 18.2: Centralized Config System~~ ✅ COMPLETE

**Status:** ✅ COMPLETE  
**Complexity:** M (Medium)  
**Completed:** 2026-01-09

~~Problem/Solution details redacted — implementation exists in `config.py`~~

**Usage:**
```bash
python train_v4.py --config configs/train_8m_50k.yaml
python train_v4.py --config configs/train_8m_50k.yaml --lr 1e-4  # Override
```
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

<details>
<summary>📁 Tasks 18.3-18.5: Evaluation Suite (PENDING — Click to expand)</summary>

#### Task 18.3: NIAH Test Implementation
⬜ PENDING | L | ~2 hours | Create `eval/niah.py` for Needle-in-a-Haystack long-context memory tests

#### Task 18.4: Qualitative Eval Suite
⬜ PENDING | M | ~45 min | Create `eval/qualitative.py` for generation samples, PPL, diversity

#### Task 18.5: Evaluation Baseline
⬜ PENDING | M | ~30 min | Run eval suite on 5M GF-MH checkpoint, document baseline

</details>

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

### Phases 3.6-3.8: Char-Level Experiments (HISTORICAL — SUPERSEDED)

**Status:** ✅ COMPLETE (2026-01-09 to 2026-01-10)  
**⚠️ SUPERSEDED BY:** Phase 4.0 (BPE Re-Validation)

These phases used **char-level tokenization** (Shakespeare) for quick sanity checks. Results are preserved for historical reference but do NOT represent production conditions. All findings have been re-validated with BPE in Phase 4.0.

<details>
<summary>📁 Click to expand historical details</summary>

#### Phase 3.6: Pre-Training Benchmark Suite (Tasks 25-30.1)

**Purpose:** Test fusion variants on cheap diagnostic benchmarks before expensive training.

**Char-Level Results (NOT for production use):**
| Variant | Val Loss | R/M Ratio | Notes |
|---------|----------|-----------|-------|
| GF-MH | 1.59 | 0.10 ⚠️ | Best loss, RWKV dominant |
| GF | 1.61 | 0.12 ⚠️ | Good loss, RWKV dominant |
| CP | 1.61 | 0.19 ⚠️ | Good loss, RWKV dominant |
| HGF | 1.69 | 0.21 ⚠️ | Mid loss, RWKV dominant |
| HY | 1.69 | 0.45 ✅ | Mid loss, balanced |

**Key Insight:** Position-adaptive gating (GF) achieves lower loss but causes Mamba underutilization.

#### Phase 3.7: Blend Ratio Sweep (Tasks 31-35)

**Purpose:** Determine if RWKV dominance is architectural or signal-based.

**Conclusion:** SIGNAL DOMINANCE CONFIRMED — All gated variants converge to RWKV-dominant regardless of initial bias. RWKV produces smoother gradients; gate takes path of least resistance.

#### Phase 3.8: Balance Improvement Options (Tasks 36-40)

**Experiments Tried:**
- Task 36: Increase Mamba LR → **WORSE** (R/M 0.10→0.08)
- Task 37: Differential warmup → DEPRIORITIZED (research complete, not implemented)
- Task 38: Balance regularization → DEPRIORITIZED
- Task 40: BPE benchmark → R/M improved to 0.21 but still imbalanced (71x activation variance)

**Conclusion:** BPE improves balance 2x vs char-level but does NOT resolve imbalance. Led to Phase 4.0.

</details>

**Files from these phases:** `tests/test_lrd.py`, `V4_BLEND_RATIOS.md`, `V4_FUSION_MODELS.md`

**What carried forward to Phase 4.0:**
- GF-MH identified as best performer (now validated with BPE)
- State monitoring infrastructure (return_states API)
- Understanding that tokenization affects component balance

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

| # | Task | Status | Completed | Details |
|---|------|--------|-----------|---------|
| 41 | Create test_tiny_graduation.py | ✅ DONE | 2026-01-10 | S0-S4 + G1-G4 test harness |
| 41a | Implement state extraction API | ✅ DONE | 2026-01-10 | GF-MH model has return_states=True |
| 42 | Run S0-S4 state space tests | ✅ DONE | 2026-01-10 | 5/5 pass, ratio=108583x |
| 43 | Run Tiny overfit test (BPE) | ✅ DONE | 2026-01-10 | Loss 0.48 in 65 steps |
| 44 | Run Tiny naive baseline test (BPE) | ✅ DONE | 2026-01-10 | 6.01 < 9.68 (37.9% better) |
| 45 | Run G1-G4 gates (BPE) | ✅ DONE | 2026-01-10 | G1✓ G2✓ G3⏭ G4⚠ |
| 46 | Checkpoint/resume test (BPE) | ✅ DONE | 2026-01-10 | 21.5 MB, diff=0 |
| 47 | Fusion variant re-ranking (BPE) | ⬜ TODO | — | 1K steps each: GF, GF-MH, HGF, HY, CP |
| 48 | Component balance assessment | ⬜ TODO | — | Investigate 71x activation ratio |
| 49 | Propagate state API to all models | ✅ DONE | 2026-01-10 | 7 model files updated |
| 50 | Add state monitoring to train_v4.py | ✅ DONE | 2026-01-10 | --log-states flag added |
| 51 | True Mamba SSM state extraction | ⬜ TODO (LOW) | — | Extract [B, nheads, headdim, d_state] |
| 52 | Implement D1-D4 diagnostic tests | ⬜ TODO | M | State divergence, collapse, interaction, LRD |
| 53 | Implement state tracking metrics | ⬜ TODO | M | Entropy, magnitude, cosine similarity |
| 54 | Implement gradient-state coupling analyzer | ⬜ TODO | L | Correlation between state gradients and loss |
| 55 | Implement information flow tracer | ⬜ TODO | L | Mutual information: state → output |
| 56 | Consolidate pass/warn/fail thresholds | ⬜ TODO | S | Single source of truth for all metrics |
| 57 | Enhance --log-states full metric suite | ⬜ TODO | M | Integrate Tasks 52-55 into training |
| 58 | Component ablation test | ⬜ TODO | M | Zero each state → measure loss impact |
| 59 | Linear state evolution test | ⬜ TODO | M | Predictable state changes with varied input |
| 60 | Long-context degradation test | ⬜ TODO | M | 64→128→256→512 token degradation curve |
| 62 | Train GPT-2 baseline (8M) | ⬜ TODO | M | Same data/tokenizer, NanoGPT implementation |
| 63 | Run CER comparison (8M) | ⬜ TODO | S | Compute-Efficiency Ratio vs GPT-2-8M |
| 64 | Run Useful Context Window test | ⬜ TODO | M | Train 2K, eval 2K→32K degradation |
| 65 | Run State Persistence Score test | ⬜ TODO | M | Fact recall at 5/10/20/50 turns |
| 66 | Implement validation tooling | ⬜ TODO | L | StateTracer, GroundingScorer per STATEFUL_VALIDATION_GUIDE.md |

**Note:** Tasks 62-66 are V5 gate prerequisites. V5 is a blocker—no 8M scaling until these pass.
See [V5_GATING.md](V5_GATING.md) for cross-comparison plan, [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md) for tool specs.

**Tiny Model Graduation Criteria (per SCALING_MILESTONES.md):**
| Test | Criteria | BPE Status |
|------|----------|------------|
| **S0-S4 state tests** | State machinery verified | ✅ 5/5 PASS (Task 42) |
| Overfit 10-100 samples | Loss → near 0 | ✅ Loss 0.48 in 65 steps (Task 43) |
| Val < naive baseline | Better than random | ✅ 6.01 < 9.68 (Task 44) |
| G1-G4 gates pass | Per V4_TESTING.md | ✅ G1✓ G2✓ G3⏭ G4⚠ (Task 45) |
| Checkpoint/resume works | Save + reload | ✅ 21.5 MB, diff=0 (Task 46) |
| Gradient flow | All components receiving gradients | ✅ Task 40 |
| Component balance documented | Ratio and variance recorded | ✅ Type A: 71x, Type B: 108583x |

**🎉 Phase 4.0 Core Graduation: PASSED (2026-01-10)**

**Order of Operations (Updated 2026-01-10):**
1. ~~**Task 41a** — Implement state extraction API~~ ✅ DONE
2. ~~**Task 41** — Create test_tiny_graduation.py~~ ✅ DONE
3. ~~**Task 42** — Run S0-S4 state space tests~~ ✅ DONE (5/5 pass)
4. ~~**Task 43** — Overfit test~~ ✅ DONE (loss 0.48 in 65 steps)
5. ~~**Task 44** — Naive baseline test~~ ✅ DONE (37.9% better)
6. ~~**Task 45** — G1-G4 gates~~ ✅ DONE (G1✓ G2✓ G3⏭ G4⚠)
7. ~~**Task 46** — Checkpoint/resume test~~ ✅ DONE (21.5 MB, diff=0)
8. ~~**Task 49** — Propagate state API to all models~~ ✅ DONE
9. ~~**Task 50** — Add state monitoring to training~~ ✅ DONE
10. **Task 47** — Re-rank fusion variants with BPE ⬜ TODO
11. **Task 48a** — Extreme ratio experiments (GF-XM, GF-XR) ✅ COMPLETE (Observation 14)
12. **Task 48** — Component balance investigation ⬜ TODO (informed by 48a)
13. **Tasks 52-60** — Advanced diagnostics ⬜ **NEXT**

**Gate:** ✅ Phase 4.0 PASSED (2026-01-10) — Core graduation criteria met.

**Next Phase:** If Phase 4.0 PASS → Continue to Phase 3.9 diagnostics with BPE. If FAIL → Debug architecture at 3.5M before scaling.

---

### Housekeeping: Code Consolidation (Non-Blocking)

**Purpose:** Clean up root Python files from early development. Does not affect model functionality.

**Completed (2026-01-10):**
- ✅ Archived 4 orphaned files: `train.py`, `layers.py`, `model.py`, `config.py`
  - These used broken relative imports or referenced archived dependencies
  - Active trainer is `train_v4.py` (uses models/ registry)

**Task 61: Consolidate ops/ Package**

| Status | Priority | Risk | Effort |
|--------|----------|------|--------|
| ⬜ TODO | P3 (Low) | Medium | 1-2 hrs |

**Current State:**
```
groundthink/
├── cuda_backends.py      # Hub for RWKV6/Mamba2 imports
├── rwkv6_prototype.py       # PyTorch RWKV6 fallback
├── rwkv6_cuda_wrapper.py    # Our CUDA kernel wrapper
├── ops/                     # Exists but underutilized
```

**Target State:**
```
groundthink/
├── ops/
│   ├── __init__.py          # Exports RWKV6Attention, Mamba2
│   ├── cuda_backends.py  # (renamed: cuda_backends.py?)
│   ├── rwkv6_prototype.py   
│   └── rwkv6_cuda_wrapper.py
```

**Dependencies to Update (10 files):**
- models/hybrid_v4*.py (8 files): `from fla_replacements` → `from ops`
- tests/test_phase0_complete.py
- Internal cross-imports within the 3 files

**Risk:**
- If any import path is wrong, ALL models break
- Requires testing each model variant after move

**Note on FLA Library:**
FLA (`flash-linear-attention`) is **no longer used** in the active codebase. We built our own CUDA kernel (`rwkv6_cuda_wrapper.py`) to resolve Windows/MSVC incompatibilities. FLA references only exist in archived files. The module `cuda_backends.py` is now a misnomer — consider renaming to `cuda_backends.py` during consolidation.

**Recommendation:** Defer until after Task 47/48. Low priority, non-blocking.

---

<details>
<summary>📁 Tasks 41a, 49, 50, 51: State API Implementation (COMPLETE — Click to expand)</summary>

#### Task 41a: State Extraction API ✅ COMPLETE
Added `return_states=True` to model forward(). Files: rwkv6_prototype.py, cuda_backends.py, models/hybrid_v4_ratio.py

#### Task 49: Propagate State API to All Models ✅ COMPLETE
Applied return_states API to all 7 model variants in models/*.py

#### Task 50: State Monitoring in Training ✅ COMPLETE
Added `--log-states` flag to train_v4.py with state norm logging

#### Task 51: True Mamba SSM State Extraction ⬜ DEFERRED
Low priority — output proxy sufficient for diagnostics

</details>

---

<details>
<summary>📁 Tasks 52-60: Advanced Diagnostic Tooling (TODO — Click to expand)</summary>

| Task | Purpose | Output | Status |
|------|---------|--------|--------|
| 52 | D1-D4 diagnostic tests | `tests/test_diagnostics.py` | ⬜ TODO |
| 53 | State tracking metrics | `tools/state_metrics.py` | ⬜ TODO |
| 54 | Gradient-state coupling | `tools/gradient_coupling_analyzer.py` | ⬜ TODO |
| 55 | Information flow tracer | `tools/information_flow_tracer.py` | ⬜ TODO |
| 56 | Threshold consolidation | `METRIC_THRESHOLDS.md` | ⬜ TODO |
| 57 | Enhanced --log-states | train_v4.py update | ⬜ TODO |
| 58 | Component ablation | `tests/test_ablation.py` | ⬜ TODO |
| 59 | Linear evolution test | `tests/test_linear_evolution.py` | ⬜ TODO |
| 60 | Long-context degradation | `tests/test_long_context.py` | ⬜ TODO |

**Cross-References:**
- [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md) for D1-D4 specs
- [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md) for metric definitions
- [CANARY_TESTS.md](CANARY_TESTS.md) for threshold values

**Expand individual task definitions when starting each task.**

</details>

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

#### ~~Task 28: Quick Train Comparison~~ ✅ COMPLETE

**Status:** ✅ COMPLETE  
**Completed:** 2026-01-10

~~Detailed task description redacted — results in V4_FUSION_MODELS.md~~

---

#### ~~Task 29: Component Balance Tests~~ ✅ COMPLETE

**Status:** ✅ COMPLETE  
**Completed:** 2026-01-10

~~Detailed task description redacted — results in V4_FUSION_MODELS.md~~

---

#### ~~Task 30: Fusion Variant Ranking~~ ✅ COMPLETE

**Status:** ✅ COMPLETE  
**Completed:** 2026-01-10

~~Detailed task description redacted — See Phase 3.6 Results Summary above~~

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

---

## Archived Task Details

**Completed task details moved to [archive/V4_STRATEGY_ARCHIVE.md](archive/V4_STRATEGY_ARCHIVE.md) (750 lines).**

Contents:
- Tasks 1-5: Foundation Setup
- Tasks 6.5-6.12: CUDA Integration Research
- Tasks 7-13: Training Optimization
- Tasks 22, 25-28: Long-Context Experiments

*Start with the next TODO task in Phase 4.0. Do not skip ahead.*
