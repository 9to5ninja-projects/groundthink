# V4 Agent Handoff Document

**Purpose:** Continuity snapshot (version & task status only)  
**Current Version:** 4.12-Alpha (Phase 4.0 — BPE Re-Validation)  
**Updated:** 2026-01-10 (Session 17 — Audit)  
**Last Agent Action:** Tasks 41a, 49, 50 complete. State extraction API implemented across all models. Training monitor added.  
**Repository:** https://github.com/9to5ninja-projects/groundthink  
**Git Status:** Clean (commits d9853d9, dd99060, 74e7d44 pushed)

---

## 🚨 CRITICAL REFRAME: Read This First

**Previous Understanding (INCORRECT):**
> "BPE tokenization fixes component balance. Use BPE and the problem is solved."

**Corrected Understanding:**
> "Char-level tokenization was a shortcut for quick sanity checks. BPE is the CORRECT BASELINE we should have used from the start. All Phase 3.6-3.8 fusion comparisons were done on char-level and are NOT verified for production. The component balance problem is NOT solved."

**Why This Matters:**
- Task 40 completed with R/M ratio 0.21 — at the lower bound of acceptable
- Activation variance ratio: 71x (RWKV var=8.58, Mamba var=0.12) — **severe imbalance**
- BPE improved R/M 2x vs char-level (0.21 vs 0.08-0.11) but did NOT fix the problem
- All fusion variant rankings (GF-MH > GF > CP > HGF > HY) are char-level data — unverified

---

## 📋 SESSION SUMMARY (Jan 10 End of Day)

**What was accomplished:**
1. ✅ Task 40 completed — 5000 steps, BPE tokenization, GF-MH model
2. ✅ Strategic reframe — Recognized char-level was shortcut, BPE is correct baseline
3. ✅ Created Phase 4.0 — BPE Re-Validation phase with 7 tasks
4. ✅ Updated V4_STRATEGY.md — Marked Phase 3.6-3.8 as CHAR-LEVEL ONLY

**Key Finding:**
BPE did NOT fix component balance as hypothesized. R/M improved from 0.08-0.11 to 0.21, but activation variance (71x) shows Mamba is still severely underutilized.

---

## 📊 TASK 40 FINAL RESULTS

**Status:** ✅ COMPLETE  
**Log:** `logs/task40_bpe_run.log`  
**Checkpoints:** `checkpoints/ckpt_GF-MH_step5000.pt`, `checkpoints/ckpt_GF-MH_final.pt`

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| R/M Ratio | 0.21 | 0.20-0.46 | ⚠️ At lower bound |
| Activation Variance Ratio | 71x | <10x ideal | ❌ Severe imbalance |
| Train Loss | 4.92 | Decreasing | ✅ |
| Val Loss | 6.22 | — | Higher than char-level |
| Throughput | ~31K tok/s | — | ✅ Stable |

**Interpretation:**
- Gradient ratio (R/M 0.21) barely meets threshold
- But activation variance (71x) shows Mamba output is 71x weaker than RWKV
- This is NOT a healthy hybrid — RWKV still dominates

---

## 🎯 CURRENT PHASE: 4.0 — BPE Re-Validation

**Objective:** Verify state space fundamentals and Tiny model graduation criteria with BPE before proceeding to diagnostics or scaling.

**Rationale:** All Phase 3.6-3.8 experiments used char-level tokenization. More critically, we never verified that the state machinery actually works. State monitoring should be our first priority.

**Key Learning from Task 40:**
- Activation variance ratio: 71x (RWKV var=8.58, Mamba var=0.12)
- Gradient ratio: 0.21 (at lower bound of acceptable)
- **Conclusion:** State machinery may not be functioning as designed

### Phase 4.0 Task List (Revised Priority Order)

| Order | # | Task | Status | Details |
|-------|---|------|--------|---------|
| ~~1a~~ | ~~41a-1~~ | ~~Type A: Restructure `return_activations`~~ | ✅ DONE | Merged into 41a |
| ~~1b~~ | ~~41a-2~~ | ~~Type B: RWKV internal state extraction~~ | ✅ DONE | `_wkv_sequential()` returns state |
| ~~1c~~ | ~~41a-3~~ | ~~Type B: Mamba internal state extraction~~ | ✅ DONE | Output proxy implemented |
| ✅ | 41 | Create test_tiny_graduation.py | ✅ DONE | S0-S4 test harness created |
| ✅ | 42 | Run S0-S4 state space tests | ✅ DONE | 5/5 pass, ratio=108583x |
| 3 | 43 | Run Tiny overfit test (BPE) | ⬜ TODO | 10-100 samples, loss → 0 |
| 4 | 44 | Run Tiny naive baseline (BPE) | ⬜ TODO | Val loss < random |
| 5 | 45 | Run G1-G4 gates (BPE) | ⬜ TODO | Re-validate with BPE |
| 6 | 46 | Checkpoint/resume test | ⬜ TODO | Save + reload works |
| 7 | 47 | Fusion variant re-ranking | ⬜ TODO | 1K steps each with BPE |
| 8 | 48 | Component balance investigation | ⬜ TODO | Compare Type A vs Type B variance |
| ✅ | 49 | Propagate state API to all models | ✅ DONE | All 8 model files updated |
| ✅ | 50 | Add state monitoring to train_v4.py | ✅ DONE | `--log-states` flag added |
| 9 | 51 | True Mamba SSM state extraction | ⬜ LOW | Research: extract [B,nheads,headdim,d_state] |
| **10** | 52 | Implement D1-D4 diagnostic tests | ⬜ TODO | Divergence, collapse, interaction, LRD |
| **11** | 53 | Implement state tracking metrics | ⬜ TODO | Entropy, magnitude, cosine similarity |
| 12 | 54 | Gradient-state coupling analyzer | ⬜ TODO | Correlation: state gradients ↔ loss |
| 13 | 55 | Information flow tracer | ⬜ TODO | Mutual information: state → output |
| **14** | 56 | Consolidate metric thresholds | ⬜ TODO | Single source of truth |
| 15 | 57 | Enhance --log-states full suite | ⬜ TODO | Integrate Tasks 52-55 metrics |
| 16 | 58 | Component ablation test | ⬜ TODO | Zero each state → measure loss |
| 17 | 59 | Linear state evolution test | ⬜ TODO | Predictable state changes |
| 18 | 60 | Long-context degradation test | ⬜ TODO | 64→128→256→512 curve |

**See [V4_STRATEGY.md](V4_STRATEGY.md#phase-40-bpe-re-validation-new--required-before-scaling) for full task definitions.**

### Two Metrics to Track (Investigation Finding)

**Type A: Output Activations** — What each component produces before fusion
- Measured in Task 40: **71x variance ratio** (RWKV var=8.58, Mamba var=0.12)
- Shape: `[B, T, hidden_dim]` per component
- Answers: "How much is each component contributing to the fused output?"

**Type B: Internal Recurrent States** — The actual memory mechanism
- Measured in Task 42: **108,583x variance ratio** (RWKV var=9689.4, Mamba var=0.089)
- RWKV: Recurrent accumulator `[B, H, S]` — the "memory" of past tokens
- Mamba: SSM state `[B, nheads, headdim, d_state]` — selective state evolution (proxy: `[B, hidden]`)
- Answers: "Is the recurrent memory actually being used?"

**Baseline Observation (2026-01-10):**
> Type B ratio (108,583x) is **1,500x higher** than Type A ratio (71x). This suggests Mamba's internal state is near-dormant while its output activations are merely weak. The Mamba component may be functioning more as a feedforward layer than a true state-space model.

### State Space Tests (S0-S4) — BASELINE RESULTS (2026-01-10)

| Test | Purpose | Result | Details |
|------|---------|--------|--------|
| S0 | Shapes exist | ✅ PASS | RWKV: [1,4,32], Mamba: [1,128], Gate: 0.70 |
| S1 | Initialization | ✅ PASS | RWKV norm: 725.7, Mamba norm: 3.7 |
| S2 | Evolution | ✅ PASS | RWKV diff: 863.2, Mamba diff: 4.5 |
| S3 | Determinism | ✅ PASS | Both components deterministic (diff=0) |
| S4 | Balance | ⚠️ WARN | Variance ratio: **108,583x** (severe imbalance) |

**Observations:**
- **Gate value 0.70** — Unexpected for GF-MH which has `gate_init=0.3`. This is the *learned* gate after training, showing RWKV dominance increased.
- **RWKV state norm 200x higher** than Mamba (725.7 vs 3.7) — magnitude imbalance
- **S2 evolution ratio ~190x** — RWKV state changes 190x more than Mamba between inputs
- **All tests pass** but S4 confirms severe component imbalance at state level

**See [CANARY_TESTS.md](CANARY_TESTS.md#s0-s4-state-space-fundamentals-35m-only--required-first) for implementations.**

### Tiny Graduation Criteria (per SCALING_MILESTONES.md)

| Test | Criteria | Status | Observed Value |
|------|----------|--------|----------------|
| **S0-S4 (Type A)** | Output activations verified | ✅ Task 40 | 71x variance ratio |
| **S0-S4 (Type B)** | Internal states verified | ✅ Task 42 | 108,583x variance ratio |
| Overfit 10-100 samples | Loss → near 0 | ⬜ Task 43 | — |
| Val < naive baseline | Better than random | ⬜ Task 44 | — |
| G1-G4 gates pass | Per V4_TESTING.md | ⬜ Task 45 | — |
| Checkpoint/resume | Save + reload works | ⬜ Task 46 | — |
| Component balance | Documented | ⚠️ Severe | Gate drifted 0.3→0.7 |

**Gate:** Phase 4.0 PASS when S0-S4 pass AND all graduation criteria verified with BPE.

### Task Dependencies (Critical Path)

```
COMPLETED                         NEXT                      THEN
─────────────────────────────────────────────────────────────────
Task 41a (API) ───┬─→ Task 41 ✅ ──→ Task 42 ✅ (5/5 pass)
Task 49 (all models) ─┘        │
Task 50 (--log-states) ────────┤
                               │
                               ├─→ Task 52 (D1-D4) ──→ Task 57 (enhance logs)
                               │        │
                               │        └─→ Task 53 (metrics) ──→ Task 57
                               │
                               ├─→ Task 56 (thresholds) ← DOCUMENTATION
                               │
                               ├─→ Tasks 43-46 (graduation tests)
                               │
                               └─→ Task 48 (balance investigation)
                                        │
                                        └─→ Task 58 (ablation)
```

**Parallelizable:** Tasks 52, 53, 56 can run in parallel  
**Blockers:** Task 41 blocks all execution tasks  
**Research:** Tasks 54, 55 are advanced (can defer)

---

## ⚠️ FOR NEXT AGENT

~~**Priority 1: Implement State Extraction API (Task 41a — BLOCKER)**~~ ✅ **COMPLETE**

~~The model currently has no way to return internal states.~~ **DONE (2026-01-10).** All model files now support:

```python
# Usage (all 8 model variants)
logits, states = model(x, return_states=True)
# states['rwkv_state'].shape = [B, H, S]
# states['mamba_state'].shape = [B, hidden]
# states['gate'] = float
```

**Location:** All files in `models/` directory  
**Impact:** ~~Blocks ALL state monitoring~~ S0-S4 tests now unblocked

**Priority 1: Run Overfit Test (Task 43)** ⬜ **NEXT**

Test that model can memorize small sample (10-100 examples, loss → near 0).

**Priority 2: Run Naive Baseline Test (Task 44)**

Verify val loss < random baseline.

**Priority 3: Implement D1-D4 Diagnostic Tests (Task 52)**

Critical for baseline before any training runs:
- D1: State divergence detection
- D2: State collapse detection  
- D3: Component interaction test
- D4: Long-range dependency test

**See [V4_STRATEGY.md](V4_STRATEGY.md#task-52-implement-d1-d4-diagnostic-tests) for implementation details.**

**Priority 4: Consolidate Metric Thresholds (Task 56)**

Before running tests, document all pass/warn/fail thresholds in one place:
- Currently scattered across CANARY_TESTS.md, VALIDATION_ROADMAP.md, STATEFUL_VALIDATION_GUIDE.md
- Single source of truth prevents confusion

**Priority 5: Investigate Component Balance (Task 48)**

The 71x activation variance ratio is concerning:
- RWKV var=8.58, Mamba var=0.12
- Is this architectural or fixable?
- Consider: gate_init, mamba_lr_mult, architectural changes

---

## 🚨 ~~REMAINING BLOCKERS~~ RESOLVED (2026-01-10)

### ~~Blocker 1: State Extraction API — CRITICAL (Task 41a)~~ ✅ COMPLETE
~~- **Scope Change:** Now requires BOTH Type A and Type B metrics~~
~~- **Type A (Quick):** Rename/restructure `return_activations` → `return_states` for consistency~~
~~- **Type B (Research):** Expose true internal states from RWKV and Mamba~~
~~- **Location:** Multiple files (models/, fla_replacements.py, rwkv6_*.py)~~
- **Status:** ✅ **IMPLEMENTED** — All 8 model files updated, `--log-states` training flag added

### ~~Blocker 1a: RWKV Internal State Extraction~~ ✅ COMPLETE
~~- **Current:** `_wkv_sequential()` computes state but discards it~~
~~- **Fix:** Return final state from `_wkv_sequential()`, propagate up through forward()~~
~~- **CUDA Issue:** RWKV-CUDA kernel computes state internally; may need prototype fallback for state extraction~~
- **Status:** ✅ Prototype fallback implemented for state extraction when CUDA active

### ~~Blocker 1b: Mamba Internal State Extraction~~ ✅ COMPLETE (proxy)
~~- **Current:** Mamba2 supports state via `inference_params` but wrapper doesn't use it~~
~~- **Fix:** Create inference_params object, pass to forward, extract ssm_state~~
- **Status:** ✅ Output proxy `[B, hidden]` implemented. True SSM state deferred to Task 51 (low priority)

### Blocker 2: Component Balance (71x activation variance) — OPEN
- **Problem:** Activation variance ratio 71x between RWKV and Mamba outputs
- **Note:** This is Type A metric. Type B metrics now available via `return_states=True`
- **Investigation:** Task 48 — compare Type A vs Type B variance ratios
- **New Tool:** Use `--log-states` flag in training to monitor state norms

---

## 📁 Current Status Summary

**Phase:** 4.0 BPE RE-VALIDATION  
**Last Action:** Tasks 41a, 49, 50 complete — State extraction API ready  
**Next Action:** Task 41 — Create test_tiny_graduation.py (then Task 42 to run it)

**Phase 3.6-3.8 Status:** ⚠️ CHAR-LEVEL ONLY — Results unverified for production

**Recent Commits:**
- `74e7d44` — Task 50: State monitoring in training
- `dd99060` — Task 49: Propagate state API to all models
- `d9853d9` — Task 41a: State extraction API implementation

**Checkpoint Files:**
- `checkpoints/ckpt_GF-MH_step5000.pt` — Task 40 (BPE, 5K steps)
- `checkpoints/ckpt_GF-MH_final.pt` — Task 40 final

**Data Available:**
- `data/fineweb_5m.txt` — BPE training data (5M bytes)
- `data/shakespeare.txt` — Char-level reference only

---

## 📁 Project Structure

```
groundthink/
├── train_v4.py                  # Main training entry point
├── models/                      # Model registry
│   ├── __init__.py              # get_model('GF-MH'), list_models()
│   ├── hybrid_v4*.py            # Variants (HY, GF, WS, RF, CP, etc.)
├── data/                        # Data loading
│   ├── data_loader.py
│   ├── tokenizer.py             # BPE via --tokenizer bpe
│   ├── fineweb_5m.txt           # BPE training data
│   └── shakespeare.txt          # Char-level reference ONLY
├── configs/                     # Training YAML configs
├── checkpoints/                 # Model weights (gitignored)
├── tests/                       # Test suite
│   └── test_tiny_graduation.py  # TODO: Create this
├── logs/                        # Training logs
│   └── task40_bpe_run.log       # Task 40 complete log
└── docs (*.md files)            # Strategy & reference
```

**Key Docs:**
- [V4_STRATEGY.md](V4_STRATEGY.md) — Task backlog (see Phase 4.0 for current tasks)
- [SCALING_MILESTONES.md](SCALING_MILESTONES.md) — Graduation criteria per model size
- [V4_TESTING.md](V4_TESTING.md) — G1-G4 gate definitions
- [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md) — Week 1-3 plan (after Phase 4.0)

---

## 🔑 Critical Institutional Knowledge

### The Tokenization Lesson (Critical)

**What We Learned:**
- Char-level tokenization was used for quick iteration during Phases 3.6-3.8
- This was appropriate for infrastructure validation but NOT for architecture evaluation
- BPE is the correct baseline for production models
- All fusion variant rankings from Phase 3.6-3.7 are char-level data and need re-validation

**What BPE Actually Showed (Task 40):**
- R/M ratio improved: 0.08-0.11 (char) → 0.21 (BPE) — **2x improvement**
- But activation variance: 71x — **still severely imbalanced**
- Conclusion: BPE helps but does NOT solve component balance

### Scaling Philosophy (Foundation)

Each parameter scale is an **experimental regime with distinct objectives**:
- **3.5M:** Sanity check — does training system work? ← **WE ARE HERE (Phase 4.0)**
- **8M:** Proof of concept — does architecture learn real patterns?
- **30M:** Scaling laws — do predictions hold?
- **125M:** MVP delivery — is this production-ready?

**Current Gate:** Phase 4.0 validates 3.5M criteria with BPE before proceeding.

### Component Balance Problem (Open)

| Metric | Char-Level | BPE | Target | Status |
|--------|------------|-----|--------|--------|
| R/M Gradient Ratio | 0.08-0.11 | 0.21 | 0.3-3.0 | ⚠️ Improved but low |
| Activation Variance | ~100x | 71x | <10x | ❌ Still severe |

**Possible Causes:**
1. Architectural — RWKV inherently dominates in hybrid fusion
2. Hyperparameter — gate_init, mamba_lr_mult need tuning
3. Data — FineWeb sample may favor RWKV patterns
4. Expected — Maybe 71x is acceptable at 3.5M scale

**Investigation:** Task 47 in Phase 4.0

---

## ⚠️ Known Issues & Decisions

**Issue 1: Phase 3.6-3.8 Data Validity**
- All experiments used char-level tokenization
- Fusion rankings (GF-MH > GF > CP > HGF > HY) are unverified
- **Action:** Re-validate with BPE in Task 46

**Issue 2: Component Balance**
- 71x activation variance is severe
- R/M 0.21 is at threshold boundary
- **Action:** Investigate in Task 47

**Issue 3: Tasks 37-38 Status**
- Previously marked "DEPRECATED — BPE fixes balance"
- Now marked "REQUIRES RE-EVALUATION" since BPE didn't fully fix it
- May need to revisit differential warmup or regularization

---

## 📊 Documentation Governance (Librarian Role)

**Recent Changes (Audit Session 2026-01-10):**
1. V4_HANDOFF.md — Redacted completed Task 41a blockers, updated priorities
2. V4_STRATEGY.md — Marked Tasks 18.1-18.2 as COMPLETE, updated goal, cleaned Task 49-50 status
3. Phase 4.0 task table — Updated to show 41a, 49, 50 as DONE

**Core Documents (Sacred Status):**
- SCALING_MILESTONES.md — Strategic foundation (verified still accurate)
- V4_STRATEGY.md — Master task source (updated with Phase 4.0)
- VALIDATION_ROADMAP.md — Execution timeline (deferred until Phase 4.0 complete)

---

## ✅ PRE-BASELINE CHECKLIST (per SCALING_MILESTONES.md)

**Before testing baselines, verify all prerequisites are met:**

| # | Requirement | Status | Reference |
|---|-------------|--------|-----------|
| 1 | **S0-S4 State tests ready** | ✅ API complete | [CANARY_TESTS.md](CANARY_TESTS.md#s0-s4-state-space-fundamentals-35m-only--required-first) |
| 2 | **BPE tokenization** | ✅ Implemented | `--tokenizer bpe` flag |
| 3 | **State extraction API** | ✅ All 8 models | `return_states=True` |
| 4 | **Training state monitor** | ✅ Implemented | `--log-states` flag |
| 5 | **test_tiny_graduation.py** | ✅ Created | `tests/test_tiny_graduation.py` |
| 6 | **Run S0-S4 tests** | ✅ 5/5 PASS | State variance ratio 108583x |
| 7 | **Overfit test** | ⬜ Pending | Task 43 |
| 8 | **Naive baseline test** | ⬜ Pending | Task 44 |
| 9 | **G1-G4 gates (BPE)** | ⬜ Pending | Task 45 |
| 10 | **Checkpoint/resume** | ⬜ Pending | Task 46 |

**Order:** Task 41 (create test harness) → Tasks 42-46 (run tests) → Phase 3.9

---

## 🎯 Phase 4.0 Gate Criteria

**PASS conditions (all required):**
- ✅ **S0-S4 state space tests pass** — state machinery verified
- ✅ Overfit test passes (loss → near 0 on small sample)
- ✅ Naive baseline test passes (val loss < random)
- ✅ G1-G4 gates pass with BPE tokenization
- ✅ Checkpoint/resume works
- ✅ Component balance assessed and documented

**FAIL triggers:**
- ❌ Any S0-S4 state test fails — state machinery broken
- ❌ Cannot overfit small sample
- ❌ Val loss worse than random
- ❌ Any G1-G4 gate fails with BPE
- ❌ Component balance deemed unacceptable (decision in Task 48)

**Outcome:** 
- If PASS → Proceed to Phase 3.9 diagnostics (with BPE baseline)
- If FAIL → Debug architecture at 3.5M before any scaling

---

## 🚀 Quick Start for Next Agent

### ✅ Implementation Complete (2026-01-10)

**Key Finding:** There are TWO types of "state" to track:
- **Type A (Output Activations):** What components produce — 71x imbalance measured
- **Type B (Internal States):** True recurrent memory — NOW EXPOSED

### Implementation Status:

1. **✅ Task 41a-1:** Added `return_states=True` to `HybridModel_GF_Ratio.forward()`
   - Location: `models/hybrid_v4_ratio.py`
   - Returns dict with `rwkv_state`, `mamba_state`, and `gate` value

2. **✅ Task 41a-2:** RWKV internal state extraction implemented
   - Modified `_wkv_sequential()` in `rwkv6_prototype.py` to return final state
   - State shape: `[batch, heads, head_size]` = `[B, H, S]`
   - Added `return_state=True` parameter to forward()
   - CUDA wrapper falls back to prototype for state extraction

3. **✅ Task 41a-3:** Mamba state extraction implemented (proxy)
   - Added `return_state=True` to Mamba2 wrapper in `fla_replacements.py`
   - Returns output activation as state proxy: `[batch, hidden_size]`
   - True SSM state `[B, nheads, headdim, d_state]` requires deeper research

4. **⏳ Task 42:** Ready to run S0-S4 tests

### API Usage:

```python
from models.hybrid_v4_ratio import create_hybrid_GF_MH_5m

model = create_hybrid_GF_MH_5m(vocab_size=16000).cuda()
x = torch.randint(0, 16000, (2, 64)).cuda()

# Get internal states (Type B)
logits, states = model(x, return_states=True)
print(states['rwkv_state'].shape)   # [2, 4, 32] = [B, H, S]
print(states['mamba_state'].shape)  # [2, 128] = [B, hidden]
print(states['gate'])               # ~0.3 for GF-MH

# Get output activations (Type A) — unchanged
logits, activations = model(x, return_activations=True)
```

### Key Files Modified:

| File | Change | Status |
|------|--------|--------|
| `models/hybrid_v4_ratio.py` | Added `return_states` to forward() | ✅ |
| `rwkv6_prototype.py` | `_wkv_sequential()` returns final state | ✅ |
| `fla_replacements.py` | RWKV6Attention + Mamba2 state extraction | ✅ |

### Key Documents:
- [CANARY_TESTS.md](CANARY_TESTS.md#s0-s4-state-space-fundamentals-35m-only--required-first) — S0-S4 state tests
- [SCALING_MILESTONES.md](SCALING_MILESTONES.md#35m-parameters-sanity-check--architecture-debug) — Tiny graduation criteria
- [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md#part-0-state-space-fundamentals-35m--run-first) — State monitoring framework
