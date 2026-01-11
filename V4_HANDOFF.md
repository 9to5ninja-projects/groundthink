# V4 Agent Handoff

**Version:** 5.0-Alpha | **Phase:** 0.5 PLANNING (Harmonization) | **Updated:** 2026-01-11

---

## Current Status

✅ **Phase 4.0 Graduation PASSED** — GF-MH passes all validation gates.
✅ **Task 62 COMPLETE** — GPT-2 baseline comparison on WikiText-103 with BPE.
✅ **Tasks 55-60 COMPLETE** — Diagnostic tooling suite built.
✅ **Librarian Audit COMPLETE** — Gap analysis between V4 and V0.5 performed.

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

**Librarian Audit & Harmonization:**
- Reconciled V4 diagnostic findings with V0.5 "Twin Debate" architecture.
- Identified critical "missed implementations" (GRUs, Qualia Fade, Semantic Weighting Sensors).
- Established new documentation strategy: Incremental conversion of research paper to tasks.
- Consolidated `ops/` package with core CUDA and prototype wrappers.

**Diagnostic Tooling Results:**
- **Mamba Paradox:** Confirmed Mamba gradients are 10x larger vs RWKV, but state contribution is <0.3%.
- **Attractor Zone:** Verified all gate initializations gravitate toward 10-30% R/M ratio.
- **BPE Efficacy:** Confirmed 16K BPE significantly improves component balance over char-level.

---

## Next Actions (V0.5 Strategy)

### Phase 0: Base Model Characterization (CURRENT PRIORITY)
| Priority | Task | Description | Status |
|----------|------|-------------|--------|
| **0.0.1** | Pure RWKV-6 Benchmark | 4M params, WikiText-103, BPE 16K | ⬜ TODO |
| **0.0.2** | Pure Mamba-2 Benchmark | 4M params, WikiText-103, BPE 16K | ⬜ TODO |
| **0.0.3** | GPT-1 Baseline | 4M params for fair comparison | ⬜ TODO |
| **0.0.4** | Comparative Analysis | Document findings, inform fusion design | ⬜ TODO |

**Rationale:** Understand individual pathway behavior before implementing fusion.

See [BASE_MODEL_CHARACTERIZATION.md](BASE_MODEL_CHARACTERIZATION.md) for detailed plan.

### Phase A: Documentation Cleanup (Sonnet) ✅ COMPLETE
| Priority | Task | Description | Status |
|----------|------|-------------|--------|
| **A1** | Trim V4_STRATEGY.md | Archive completed task details, keep only summary tables. Target: <500 lines. | ✅ COMPLETE (206 lines) |
| **A2** | Consolidate V4 Docs | Merge redundant findings into OBSERVATION_SYNTHESIS.md. | ✅ COMPLETE (archived 2 docs) |
| **A3** | Finalize V0.5_ROADMAP.md | Ensure all Section 1 tasks have clear acceptance criteria. | ✅ COMPLETE (6 tasks defined) |
| **A4** | Update DOCUMENTATION_MAP.md | Reflect new file structure (V0.5 docs, archived V4 details). | ✅ COMPLETE (navigation updated) |

### Phase B: Implementation (Sonnet) - READY TO START
| Priority | Task | Description |
|----------|------|-------------|
| **B1** | GRU Arbiter (Task 0.1) | Replace `nn.Linear` gate with `nn.GRUCell` in `ops/`. |
| **B2** | Mamba Residual (Task 0.2) | Add `h = x + mamba(x)` skip connection. |
| **B3** | Debate Loss (Task 0.3) | Implement cosine similarity penalty in `tools/`. |
| **B4** | Pilot Run (Task 0.4) | 5K steps, verify Mamba contribution > 5%. |

**✅ Phase A Gate: PASSED** - Documentation clean, ready for implementation.

---

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
