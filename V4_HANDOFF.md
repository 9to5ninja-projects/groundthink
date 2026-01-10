# V4 Agent Handoff Document

**Purpose:** Continuity snapshot (version & task status only)  
**Current Version:** 4.11-Alpha (Phase 4.0 — BPE Re-Validation)  
**Updated:** 2026-01-10 (End of Day)  
**Last Agent Action:** Task 40 complete, strategic reframe — BPE is correct baseline, not a fix. Created Phase 4.0.  
**Repository:** https://github.com/9to5ninja-projects/groundthink  
**Git Status:** Modified (V4_STRATEGY.md, V4_HANDOFF.md updated)

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
| **1** | 41a | Implement state extraction API | ⬜ **BLOCKER** | Add `return_states` to model forward() |
| **2** | 42 | Run S0-S4 state space tests | ⬜ TODO | Verify state machinery works |
| **3** | 41 | Create test_tiny_graduation.py | ⬜ TODO | Include S0-S4 + G1-G4 |
| 4 | 43 | Run Tiny overfit test (BPE) | ⬜ TODO | 10-100 samples, loss → 0 |
| 5 | 44 | Run Tiny naive baseline (BPE) | ⬜ TODO | Val loss < random |
| 6 | 45 | Run G1-G4 gates (BPE) | ⬜ TODO | Re-validate with BPE |
| 7 | 46 | Checkpoint/resume test | ⬜ TODO | Save + reload works |
| 8 | 47 | Fusion variant re-ranking | ⬜ TODO | 1K steps each with BPE |
| 9 | 48 | Component balance investigation | ⬜ TODO | Why 71x variance? |

### State Space Tests (S0-S4) — NEW PRIORITY

| Test | Purpose | Pass Criteria |
|------|---------|---------------|
| S0 | State shapes exist | Correct dimensions |
| S1 | Initialization health | Norm 0.01-100, no NaN |
| S2 | State evolution | Different inputs → different states |
| S3 | State determinism | Same input → same state |
| S4 | Component contribution | Variance ratio <100x |

**See [CANARY_TESTS.md](CANARY_TESTS.md#s0-s4-state-space-fundamentals-35m-only--required-first) for implementations.**

### Tiny Graduation Criteria (per SCALING_MILESTONES.md)

| Test | Criteria | Status |
|------|----------|--------|
| **S0-S4 state tests** | State machinery verified | ❓ **API missing** |
| Overfit 10-100 samples | Loss → near 0 | ❓ Not tested |
| Val < naive baseline | Better than random | ❓ Not tested |
| G1-G4 gates pass | Per V4_TESTING.md | ❓ Not tested with BPE |
| Checkpoint/resume | Save + reload works | ❓ Not tested with BPE |
| Component balance | Documented | ⚠️ 71x variance (severe) |

**Gate:** Phase 4.0 PASS when S0-S4 pass AND all graduation criteria verified with BPE.

---

## ⚠️ FOR NEXT AGENT

**Priority 1: Implement State Extraction API (Task 41a — BLOCKER)**

The model currently has no way to return internal states. Implement:

```python
# In models/hybrid_v4_GF.py
def forward(self, x, return_states=False):
    # ... existing forward logic ...
    
    if return_states:
        return output, {
            'rwkv_state': rwkv_hidden,    # Internal RWKV state
            'mamba_state': mamba_hidden,  # Internal Mamba state  
            'gate_values': gate_output    # Fusion gate values
        }
    return output
```

**Location:** [models/hybrid_v4_GF.py](models/hybrid_v4_GF.py)  
**Impact:** Blocks ALL state monitoring (S0-S4, State Tracing, diagnostics)

**Priority 2: Run S0-S4 State Space Tests (Task 42)**

Once state extraction works, verify state machinery:
```bash
python tests/test_tiny_graduation.py --test-states --tokenizer bpe
```

**Priority 3: Create test_tiny_graduation.py (Task 41)**

Combine all tests:
- S0-S4 state space fundamentals
- G1-G4 validation gates
- Overfit test
- Naive baseline test
- Checkpoint/resume test

**Priority 4: Investigate Component Balance (Task 48)**

The 71x activation variance ratio is concerning:
- RWKV var=8.58, Mamba var=0.12
- Is this architectural or fixable?
- Consider: gate_init, mamba_lr_mult, architectural changes

---

## 🚨 REMAINING BLOCKERS

### Blocker 1: State Extraction API — CRITICAL (Task 41a)
- **Location:** [models/hybrid_v4_GF.py](models/hybrid_v4_GF.py)
- **Problem:** No `return_states=True` parameter to get internal states
- **Impact:** Cannot run S0-S4 state tests, State Tracing Module, or diagnostics
- **Fix:** Add return_states parameter that returns RWKV state, Mamba state, gate values
- **Status:** ⬜ **BLOCKER — FIX FIRST**

### Blocker 2: Hidden State Extraction — SUPERSEDED
- **Status:** Merged into Blocker 1 (same issue, different framing)

### Blocker 3: Component Balance (71x variance)
- **Problem:** Activation variance ratio 71x between RWKV and Mamba
- **Impact:** Mamba may not be contributing meaningfully to model output
- **Investigation:** Task 48 — need state monitoring to understand root cause

---

## 📁 Current Status Summary

**Phase:** 4.0 BPE RE-VALIDATION  
**Last Action:** Task 40 complete, strategic reframe, added S0-S4 state tests  
**Next Action:** Task 41a — Implement state extraction API (BLOCKER)

**Phase 3.6-3.8 Status:** ⚠️ CHAR-LEVEL ONLY — Results unverified for production

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

**Recent Changes (This Session):**
1. V4_STRATEGY.md — Added Phase 4.0, marked 3.6-3.8 as CHAR-LEVEL ONLY
2. V4_HANDOFF.md — Complete rewrite with corrected framing
3. Task 40 status — Updated from RUNNING to COMPLETE with results

**Core Documents (Sacred Status):**
- SCALING_MILESTONES.md — Strategic foundation (verified still accurate)
- V4_STRATEGY.md — Master task source (updated with Phase 4.0)
- VALIDATION_ROADMAP.md — Execution timeline (deferred until Phase 4.0 complete)

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

1. **Read the CRITICAL REFRAME section at the top** — understand the strategic shift
2. **Fix Blocker 1 (Task 41a)** — implement `return_states` in model forward()
3. **Run S0-S4 state tests (Task 42)** — verify state machinery works
4. **Create test_tiny_graduation.py (Task 41)** — combine all graduation tests
5. **Investigate 71x variance (Task 48)** — understand component balance issue

**Order of Operations:**
```
Task 41a → Task 42 → Task 41 → Tasks 43-46 → Task 47 → Task 48
(API)      (S0-S4)   (Script)  (Grad tests)  (Fusion)  (Balance)
```

**Do NOT proceed to Phase 3.9 diagnostics until Phase 4.0 is complete.**

**Key Documents:**
- [CANARY_TESTS.md](CANARY_TESTS.md#s0-s4-state-space-fundamentals-35m-only--required-first) — S0-S4 state tests
- [SCALING_MILESTONES.md](SCALING_MILESTONES.md#35m-parameters-sanity-check--architecture-debug) — Tiny graduation criteria
- [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md#part-0-state-space-fundamentals-35m--run-first) — State monitoring framework
