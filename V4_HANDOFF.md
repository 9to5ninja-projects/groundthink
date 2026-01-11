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

1. Created GF-XM (gate_init=0.03) and GF-XR (gate_init=0.97) extreme ratio variants
2. Ran 500-step training + S0-S4 graduation on both
3. **Key Finding (Observation 14):** Bidirectional attractor zone ~0.06-0.27 R/M
   - GF-XM drifted 0.03→0.25 (toward RWKV)
   - GF-XR drifted 0.97→0.27 (toward Mamba)
4. Added `print_test_header()` to graduation suite for proper logging
5. Documented in V4_FUSION_MODELS.md as Observation 14

---

## Next Actions (In Order)

| Priority | Tasks | Description |
|----------|-------|-------------|
| **1** | 52-57 | Diagnostic tooling (D1-D4 tests, state tracking, gradient-state coupling) |
| **2** | 58-61 | Analysis tools (component ablation, linear state evolution, long-context) |
| **3** | 62-66 | V5 gate (GPT-2 baseline, CER, UCW, SPS, validation tooling) |

**V5 is a blocker** — no 8M scaling until Tasks 62-66 pass.
But Tasks 52-61 inform 62-66, so build diagnostics first.

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
| S4 State Variance | ⚠️ WARN | 66K-124K ratio (architecture-dependent) |
| Gate Attractor | ℹ️ INFO | All gates converge to 0.06-0.27 zone |

**Finding (Observation 14):** Optimizer finds loss-minimizing attractor regardless of init.
- GF-XM (0.03 init): 66K S4 ratio, 1.81 val loss
- GF-MH (0.30 init): 88K S4 ratio, ~1.58 val loss ← still best
- GF-XR (0.97 init): 124K S4 ratio, 1.96 val loss

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
