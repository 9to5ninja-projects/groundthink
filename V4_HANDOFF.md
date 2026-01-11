# V4 Agent Handoff

**Version:** 5.0-Alpha | **Phase:** 4.0 COMPLETE | **Updated:** 2026-01-11

---

## Current Status

✅ **Phase 4.0 Graduation PASSED** — GF-MH passes all validation gates.
✅ **Task 62 COMPLETE** — GPT-2 baseline comparison on WikiText-103 with BPE.
✅ **Tasks 55-60 COMPLETE** — Diagnostic tooling suite built.

| Test | Result |
|------|--------|
| S0-S4 State Tests | 5/5 PASS (variance ratio 108,583x) |
| Task 43 Overfit | PASS (loss 0.48 in 65 steps) |
| Task 44 Baseline | PASS (6.01 vs 9.68, 37.9% better) |
| G1-G4 Gates | G1✓ G2✓ G3⏭ G4⚠ |
| Task 46 Checkpoint | PASS (21.5 MB, identical reload) |
| **Task 62 GPT-2** | **EQUIVALENT** (ratio 1.008) |
| Task 58 Ablation | FAIL (RWKV 99.9%, Mamba 0.1%) |
| Task 59 Evolution | PASS (state responds to input) |
| Task 60 Long-ctx | PASS (1.04x ratio) |

---

## Last Session (2026-01-11)

**Tasks 55-60 COMPLETE:** Diagnostic tooling suite

| Task | File | Result |
|------|------|--------|
| 55 | tools/information_flow_tracer.py | Mamba <0.3% |
| 56 | tools/thresholds.py | Unified thresholds |
| 57 | train_v4.py | --log-states enhanced |
| 58 | tests/test_ablation.py | RWKV 99.9%, Mamba 0.1% |
| 59 | tests/test_state_evolution.py | PASS |
| 60 | tests/test_long_context.py | 1.04x PASS |

**Key finding:** Mamba contribution is essentially zero. RWKV dominates.
Long-context stable (no degradation at 512 tokens).

---

## Next Actions (In Order)

| Priority | Task | Description |
|----------|------|-------------|
| **1** | 63 | CER (Compute-Efficiency Ratio) calculation |
| **2** | 64 | UCW (Useful Context Window) test |
| **3** | 65 | SPS (State Persistence Score) at 5/10/20/50 turns |

**CRITICAL:** All V5 benchmarks must use:
- WikiText-103 data (`data/wikitext103/train.txt`)
- BPE tokenizer (`data/tokenizer_wikitext.json`, vocab=16K)
- Same data/tokenizer for both GPT-2 and GF-MH

See [V5_GATING.md](V5_GATING.md) for thresholds and criteria.

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
| [tests/test_diagnostics.py](tests/test_diagnostics.py) | D1-D4 diagnostic analysis |
| [tests/test_ablation.py](tests/test_ablation.py) | Component ablation (Task 58) |
| [tests/test_long_context.py](tests/test_long_context.py) | 64-512 degradation (Task 60) |
| [tools/thresholds.py](tools/thresholds.py) | Unified thresholds (Task 56) |
| [tools/information_flow_tracer.py](tools/information_flow_tracer.py) | MI tracing (Task 55) |
| [tools/state_metrics.py](tools/state_metrics.py) | State health tracking |
| [tools/gradient_coupling.py](tools/gradient_coupling.py) | Gradient flow analysis |

---

## Known Issues

| Issue | Status | Notes |
|-------|--------|-------|
| G4 Gradient Imbalance | ⚠️ WARN | Mamba grads 10x larger than RWKV |
| S4 State Variance | ⚠️ WARN | 66K-124K ratio (architecture-dependent) |
| D1 State Divergence | ⚠️ WARN | RWKV norm grows 2.5x over 512 tokens |
| D3 Component Balance | ⚠️ WARN | Mamba only 0.2% contribution by state norm |
| Gate Attractor | ℹ️ INFO | All gates converge to 0.06-0.27 zone |

**Finding (Observation 14):** Optimizer finds loss-minimizing attractor regardless of init.
- GF-XM (0.03 init): 66K S4 ratio, 1.81 val loss
- GF-MH (0.30 init): 88K S4 ratio, ~1.58 val loss ← still best
- GF-XR (0.97 init): 124K S4 ratio, 1.96 val loss

---

## Git Status

```
Latest: c6c2f59
Branch: main
Status: Clean (pending doc sync)
```

---

*For detailed task definitions, see [V4_STRATEGY.md](V4_STRATEGY.md)*  
*For version history, see [CHANGELOG.md](CHANGELOG.md)*
