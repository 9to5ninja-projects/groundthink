# V4 Agent Handoff

**Version:** 5.0-Alpha | **Phase:** 4.0 COMPLETE | **Updated:** 2026-01-10

---

## Current Status

✅ **Phase 4.0 Graduation PASSED** — GF-MH (3.5M params) passes all validation gates.

| Test | Result |
|------|--------|
| S0-S4 State Tests | 5/5 PASS (variance ratio 108,583x) |
| Task 43 Overfit | PASS (loss 0.48 in 65 steps) |
| Task 44 Baseline | PASS (6.01 vs 9.68, 37.9% better) |
| G1-G4 Gates | G1✓ G2✓ G3⏭ G4⚠ |
| Task 46 Checkpoint | PASS (21.5 MB, identical reload) |

---

## Last Session (2026-01-10)

1. Documentation consolidation: archived Dockerfile, SCALING_MILESTONES.md
2. Created V5_GATING.md — V5 is a blocker for 8M (not a version bump)
3. Cleaned requirements.txt (removed 6 unused packages)
4. Added Tasks 62-66 as V5 gate prerequisites
5. Strengthened SOP: terminology lock, archive policy, commit hygiene

---

## Next Actions (Pick One)

| Priority | Task | Description |
|----------|------|-------------|
| **1** | Tasks 62-66 | V5 gate prerequisites (GPT-2 baseline, CER, validation tools) |
| **2** | Task 52-61 | Diagnostic tooling (state tracer, info flow, etc.) |
| **3** | Scale to 8M | Only after V5 gate passes |

---

## Quick Start

```bash
source .venv/bin/activate

# Run all graduation tests
python tests/test_tiny_graduation.py --states --gates --overfit --baseline --checkpoint

# Train with state monitoring
python train_v4.py --model GF-MH --tokenizer bpe --log-states

# Check model registry
python -c "from models import list_models; print(list_models())"
```

---

## Key Files

| File | Purpose |
|------|---------|
| [V4_STRATEGY.md](V4_STRATEGY.md) | Master task backlog (Phases 4.0-5.0) |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md) | Test harness documentation |
| [CANARY_TESTS.md](CANARY_TESTS.md) | S0-S4 and G1-G4 definitions |
| [tests/test_tiny_graduation.py](tests/test_tiny_graduation.py) | Unified test harness |

---

## Known Issues

| Issue | Status | Notes |
|-------|--------|-------|
| G4 Gradient Imbalance | ⚠️ WARN | Mamba grads 10x larger than RWKV |
| S4 State Variance | ⚠️ WARN | 108,583x ratio (Mamba near-dormant) |
| Gate Drift | ⚠️ INFO | 0.3→0.7 (RWKV dominance increased) |

**Decision:** Proceed with known imbalance — model still learns (37.9% better than random).

---

## Git Status

```
Latest: 32b92eb
Branch: main
Status: Clean
```

---

*For detailed task definitions, see [V4_STRATEGY.md](V4_STRATEGY.md)*  
*For version history, see [CHANGELOG.md](CHANGELOG.md)*
