# V4 Agent Handoff Document

**Purpose:** Continuity snapshot (version & task status only)  
**Current Version:** 4.10-Alpha (Phase 3.8 Task 36 Complete)  
**Updated:** 2026-01-10  
**Last Agent Action:** Task 36 - Higher Mamba LR (1.0) made imbalance WORSE (R/M 0.10→0.08)  
**Repository:** https://github.com/9to5ninja-projects/groundthink  
**Git Status:** ✅ Clean (commit b8c9dbb)

---

## ⚠️ WORKFLOW (Read First)

**Mandatory on startup:**
1. Skim this document (version & status)
2. Audit [V4_STRATEGY.md](V4_STRATEGY.md) Task Assessment Matrix
3. Write todo list with `manage_todo_list` tool
4. Begin work

**Procedure reference:** See [GETTING_STARTED.md](GETTING_STARTED.md) for full workflow, environment setup, validation gates, versioning protocol.

---

## Current Status

**Phase:** 3.8 IN PROGRESS — Addressing RWKV Signal Dominance  
**Last Task:** Task 36 (Mamba LR experiment) COMPLETE

**Key Findings (Phase 3.7-3.8):**
- ✅ RWKV dominance is **signal-based** (not architectural)
- ✅ Higher Mamba LR (1.0 vs 0.5) made imbalance **worse** (R/M 0.10→0.08) 
  - Conclusion: Mamba not LR-starved; faster learning accelerates convergence to RWKV dominance
- 🔬 Next hypothesis: Differential warmup schedules (Mamba may need longer/slower warmup to learn independently)
  - RWKV warmup: 20-2500 steps per BlinkDL recommendations
  - Mamba warmup: No explicit guidance; SSM recurrent dynamics need careful initialization
  - **Proposed:** Per-group LambdaLR schedules (PyTorch supports this natively)

**Phase 3.8 Tasks (Ordered):**
- Task 36: ✅ COMPLETE (Mamba LR 1.0 experiment)
- Task 37: 🟡 Research COMPLETE, ready for implementation (differential warmup schedules)
- Task 38: ⬜ Balance regularization loss (architectural change)
- Task 39: ⬜ Accept RWKV dominance decision

**Checkpoint Files:**
- `ckpt_GF-MH_step1000.pt`, `ckpt_GF-MH_final.pt` (Task 36 run)

See [V4_FUSION_MODELS.md](V4_FUSION_MODELS.md#observations) for Phase 3.7 detailed results.

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
│   ├── tokenizer.py
│   └── shakespeare.txt
├── configs/                     # Training YAML configs
├── checkpoints/                 # Model weights (gitignored)
├── tests/                       # Test suite
├── V4_DESIGN.md                 # Architecture spec
├── V4_STRATEGY.md               # Task backlog + assessment
├── V4_TESTING.md                # Validation gates procedure
├── V4_FUSION_MODELS.md          # Technical reference (8 fusion variants)
├── V4_TRAINING_GUIDE.md         # Training reference + hyperparameters
├── fla_replacements.py          # RWKV/Mamba component bridge
└── rwkv6_*.py                   # RWKV-6 implementations
```

**Key Docs:**
- [V4_STRATEGY.md](V4_STRATEGY.md) — Task backlog with complexity ratings
- [V4_DESIGN.md](V4_DESIGN.md) — Architecture specification
- [V4_FUSION_MODELS.md](V4_FUSION_MODELS.md) — Technical reference for fusion variants
- [V4_TRAINING_GUIDE.md](V4_TRAINING_GUIDE.md) — Training procedures & hyperparameters (warmup schedules, config system, model registry)
- [V4_TESTING.md](V4_TESTING.md) — Validation gates (G1-G4)
- [GETTING_STARTED.md](GETTING_STARTED.md) — Setup, workflow, environment
- [CONTRIBUTING.md](CONTRIBUTING.md) — Git protocol, versioning

---

## Next Agent Instructions

**Before starting:** 
1. Read [GETTING_STARTED.md](GETTING_STARTED.md#-mandatory-workflow) (workflow)
2. Audit [V4_STRATEGY.md](V4_STRATEGY.md) (tasks and complexity)
3. Write todo list with `manage_todo_list` tool

**Current Focus:** Phase 3.8 Task 37 — Implement per-group warmup schedules to help Mamba learn independently

**Task 37 Details:**
- **Hypothesis:** Mamba may need longer/slower warmup period to avoid convergence to RWKV-dominant state
- **Approach:** Refactor `get_lr_lambda()` in train_v4.py to support per-group lambdas
- **Implementation:** PyTorch LambdaLR accepts `lr_lambda=[lambda1, lambda2, lambda3]` for per-group schedules
- **Experiments:**
  - 37a: Mamba 2x warmup (1000 steps vs 500 RWKV)
  - 37b: RWKV slow ramp (BlinkDL spike-fix formula)
  - 37c/d: Mamba delayed start

**Research:** Complete (documented in [V4_TRAINING_GUIDE.md](V4_TRAINING_GUIDE.md) — search "Per-Component Warmup Schedules")

**Expected Outcome:** If per-group warmup helps Mamba learn independently early on, R/M ratio should improve (target: 0.3-0.5 range)
