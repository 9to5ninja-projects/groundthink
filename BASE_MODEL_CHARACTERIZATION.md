# Base Model Characterization (V0.5)

**Version:** V0.5 Phase 0  
**Status:** PREREQUISITE for hybrid fusion development  
**Updated:** 2026-01-11  
**Purpose:** Benchmark pure RWKV-6 and Mamba-2 before fusion design

---

## Objective

Understand individual pathway behavior **before** implementing hybrid fusion. Findings inform V0.5 gating and fusion architecture decisions.

**Key Question:** What do RWKV-6 and Mamba-2 do well/poorly on their own?

---

## Test Matrix

| Model | Scale | Dataset | Tokenizer | Baseline |
|-------|-------|---------|-----------|----------|
| Pure RWKV-6 | 4M, 8M | WikiText-103 | BPE 16K | GPT-1, GPT-2 |
| Pure Mamba-2 | 4M, 8M | WikiText-103 | BPE 16K | GPT-1, GPT-2 |
| GPT-1 (baseline) | 4M, 8M | WikiText-103 | BPE 16K | Reference |
| GPT-2 (baseline) | 6.8M | WikiText-103 | BPE 16K | ✅ Complete (Task 62) |

---

## Metrics to Collect

### Primary Metrics
- [ ] Validation perplexity
- [ ] Training throughput (tok/s)
- [ ] Memory footprint (MB)
- [ ] Loss convergence curve

### State Dynamics
- [ ] State norm evolution (existing: `tools/state_metrics.py`)
- [ ] State response to input variety (existing: `tests/test_state_evolution.py`)
- [ ] Long-context degradation 64→512 tokens (existing: `tests/test_long_context.py`)

### Gradient Analysis
- [ ] Gradient magnitude per layer (existing: `tools/gradient_coupling.py`)
- [ ] Gradient flow health (no vanishing/exploding)

### Information Flow
- [ ] Mutual information state→output (existing: `tools/information_flow_tracer.py`)
- [ ] Layer-wise contribution (existing: `tests/test_ablation.py`)

---

## Tasks

### Task 0.0.1: Pure RWKV-6 Benchmark (4M)
**Goal:** Establish baseline RWKV-6 performance at 4M scale.

**Acceptance Criteria:**
- [ ] Model trains to convergence on WikiText-103
- [ ] Validation perplexity recorded
- [ ] State evolution characterized (norm growth, variance)
- [ ] Gradient health verified (no pathology)
- [ ] Compared against GPT-1 (4M) and GPT-2 (6.8M)

**Tools:** `train_v4.py --model RWKV-only --log-states`

---

### Task 0.0.2: Pure Mamba-2 Benchmark (4M)
**Goal:** Establish baseline Mamba-2 performance at 4M scale.

**Acceptance Criteria:**
- [ ] Model trains to convergence on WikiText-103
- [ ] Validation perplexity recorded
- [ ] State evolution characterized (selective update patterns)
- [ ] Gradient health verified
- [ ] Compared against GPT-1 (4M) and GPT-2 (6.8M)

**Tools:** `train_v4.py --model Mamba-only --log-states`

---

### Task 0.0.3: GPT-1 Baseline (4M)
**Goal:** Establish GPT-1 baseline for fair comparison.

**Acceptance Criteria:**
- [ ] GPT-1 implementation at 4M params
- [ ] Trained on WikiText-103 with BPE 16K
- [ ] Validation perplexity recorded
- [ ] Reference point for RWKV-6 and Mamba-2

**Tools:** `models/gpt1.py` (create), `tests/exp002_base_comparison.py`

---

### Task 0.0.4: Comparative Analysis
**Goal:** Document behavioral differences and inform fusion design.

**Acceptance Criteria:**
- [ ] Perplexity comparison table (RWKV vs Mamba vs GPT-1 vs GPT-2)
- [ ] State dynamics comparison (recurrent vs selective patterns)
- [ ] Gradient flow comparison (magnitude, stability)
- [ ] Strengths/weaknesses documented per model
- [ ] Recommendations for fusion architecture

**Output:** `BASE_MODEL_FINDINGS.md` (created after Task 0.0.4)

---

### Task 0.0.5: Scale Test (8M)
**Goal:** Validate findings hold at 8M scale.

**Acceptance Criteria:**
- [ ] Pure RWKV-6 at 8M trained
- [ ] Pure Mamba-2 at 8M trained
- [ ] Comparison against 4M results
- [ ] Scaling behavior documented

**Gate:** Only if 4M results show promise for fusion.

---

## Tools Already Available

| Tool | Purpose | Used For |
|------|---------|----------|
| `tools/state_metrics.py` | State health tracking | RWKV/Mamba state evolution |
| `tools/gradient_coupling.py` | Gradient flow analysis | Per-model gradient health |
| `tools/information_flow_tracer.py` | Mutual information | State→output contribution |
| `tests/test_state_evolution.py` | State response to input | Behavioral characterization |
| `tests/test_long_context.py` | Context degradation | Long-range capability |
| `tests/test_ablation.py` | Component zeroing | Layer-wise contribution |
| `models/gpt2.py` | GPT-2 baseline | ✅ Complete (Task 62) |

---

## Success Criteria

**Phase 0 COMPLETE when:**
- ✅ Pure RWKV-6 benchmarked at 4M (Task 0.0.1)
- ✅ Pure Mamba-2 benchmarked at 4M (Task 0.0.2)
- ✅ GPT-1 baseline at 4M (Task 0.0.3)
- ✅ Comparative analysis documented (Task 0.0.4)
- ✅ Clear recommendations for fusion architecture

**Informs:** [V0.5_ROADMAP.md](V0.5_ROADMAP.md) fusion design decisions.

---

## Timeline Estimate

| Task | Estimated Time | Notes |
|------|----------------|-------|
| 0.0.1 RWKV-6 4M | 2-3 days | Includes training + analysis |
| 0.0.2 Mamba-2 4M | 2-3 days | Includes training + analysis |
| 0.0.3 GPT-1 4M | 1-2 days | Implementation + training |
| 0.0.4 Analysis | 1 day | Document findings |
| 0.0.5 Scale test | 3-4 days | Optional, depends on 4M results |

**Total:** 1-2 weeks for Phase 0 completion.

---

## References

- **Hybrid Architecture Plan:** [V0.5_ROADMAP.md](V0.5_ROADMAP.md)
- **V4 Hybrid Results:** [OBSERVATION_SYNTHESIS.md](OBSERVATION_SYNTHESIS.md)
- **Diagnostic Tools:** [V4_TESTING.md](V4_TESTING.md)
- **GPT-2 Baseline:** Task 62 (loss ratio 1.008)
- **Research Foundation:** [groundthink_architecture_research.md](groundthink_architecture_research.md)

---

**Status:** Tasks 0.0.1-0.0.5 define Phase 0. Start with Task 0.0.1.
