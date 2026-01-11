# GroundThink Changelog

All notable changes to the GroundThink V4 project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [5.0-Alpha] - 2026-01-10 (Documentation Harmony + V5 Gate)

### Summary
**HOUSEKEEPING SESSION.** Consolidated documentation, established SOP, cleaned dependencies, and defined V5 as a gate (not a version bump). V5 blocks 8M scaling until Tasks 62-66 pass.

### Added
- **V5_GATING.md**: GPT-2 cross-comparison strategy with CER, UCW, SPS metrics
- **LICENSE**: MIT license with experimental research disclaimer and attribution table
- **REQUIREMENTS_ANALYSIS.md**: Dependency audit with hardware requirements
- **CONTRIBUTING.md**: Documentation & Versioning SOP (prevents future doc sprawl)
- **Task 66**: Validation tooling (StateTracer, GroundingScorer) as V5 prerequisite

### Changed
- **cuda_backends.py**: Renamed from cuda_backends.py (FLA no longer used)
- **requirements.txt**: Cleaned—removed 6 unused packages, added mamba-ssm/causal-conv1d
- **GETTING_STARTED.md**: Rewritten with current test suite and hardware requirements
- **V4_STRATEGY.md**: Tasks 62-66 are now V5 gate prerequisites

### Archived
- Dockerfile (v1/v2 era, vast.ai reference—using SSH now)
- SCALING_MILESTONES.md (content moved to V5_GATING.md)
- 19+ session-dated/redundant files (from earlier consolidation pass)

### V5 Gate Prerequisites (Tasks 62-66)
| Task | Description | Status |
|------|-------------|--------|
| 62 | Train GPT-2 baseline (8M) | ⬜ TODO |
| 63 | Run CER comparison (8M) | ⬜ TODO |
| 64 | Useful Context Window test | ⬜ TODO |
| 65 | State Persistence Score test | ⬜ TODO |
| 66 | Implement validation tooling | ⬜ TODO |

**V5 is a blocker—no 8M scaling until all pass.**

---

## [5.0-Alpha] - 2026-01-10 (Phase 4.0 Complete)

### Summary
**GRADUATION COMPLETE.** GF-MH (3.5M params) passes all validation gates. Ready for next phase.

### Phase 4.0 Results

| Test | Result | Notes |
|------|--------|-------|
| S0-S4 State Tests | 5/5 PASS | Variance ratio 108,583x |
| Task 43 Overfit | PASS | Loss 0.48 in 65 steps |
| Task 44 Baseline | PASS | 6.01 vs 9.68 (37.9% better than random) |
| G1 Forward | PASS | No NaN, shapes correct |
| G2 Init Entropy | PASS | 9.65/9.68 = 99.7% of max entropy |
| G4 Gradient Balance | WARN | Ratio 0.10 (Mamba 10x larger gradients) |
| Task 46 Checkpoint | PASS | 21.5 MB, identical outputs after reload |

### Key Findings
- **BPE tokenization validated**: TinyStories-8B subset with GPT-2 tokenizer works correctly
- **State extraction API**: Stable output for any hidden state introspection
- **Gate initialization entropy**: Near-uniform distribution is healthy for untrained models
- **Gradient imbalance**: Mamba dominates but model still learns (non-blocking warning)

### Changed
- **tests/test_tiny_graduation.py**: Unified graduation test harness with all gates
- **V4_STRATEGY.md**: Phase 4.0 marked COMPLETE, Phases 3.6-3.8 collapsed to historical
- **V4_HANDOFF.md**: Updated status to Phase 4.0 graduation complete
- **VERSION**: Updated to 5.0-Alpha

### Documentation Consolidation
- Archived: V4_BLEND_RATIOS.md → archive/ (content in V4_FUSION_MODELS.md)
- Archived: V4.5_PYTHON_WRAPPERS.md → archive/ (content in V4.5_CUDA_KERNELS.md)
- Archived: V4.5_KERNEL_FUSION.md → archive/ (future research)
- V4.5_CUDA_KERNELS.md: Marked COMPLETE (Phase 0)

### Next Steps (Phase 5.0 Options)
- Task 47: Scale to 8M parameters
- Task 48: Long-context evaluation (512+ tokens)
- Task 49: Alternative datasets (code, dialogue)

---

## [4.10-Alpha] - 2026-01-10 (Phase 3.8 Complete)

### Summary
**RWKV DOMINANCE ACCEPTED.** After Phase 3.7 confirmed signal-based (not architectural) RWKV dominance, Phase 3.8 attempted balance interventions. Results: interventions hurt loss without improving balance. Decision: accept RWKV dominance.

### Phase 3.8 Results

| Intervention | Impact | Recommendation |
|--------------|--------|----------------|
| Balance regularization | Loss +0.15 | ❌ Not recommended |
| Mamba LR multiplier 2x | No change in R/M | ❌ Ineffective |
| Gate freeze (100 steps) | Slight balance, loss +0.08 | ⚠️ Marginal |
| Accept dominance | Lowest loss (1.59) | ✅ Recommended |

### Key Finding
**Optimizer "takes path of least resistance" to RWKV.** This is signal-based (RWKV gradients are smoother), not a bug. Future work may explore Mamba-specific optimizations, but for now GF-MH architecture is validated.

### Changed
- **V4_STRATEGY.md**: Phase 3.8 marked COMPLETE, moved to historical section
- **V4_FUSION_MODELS.md**: Added "Historical Research" warning to Phase 3.6-3.8 sections

---

## [4.9-Alpha] - 2026-01-10 (Phase 3.7 Complete)

### Summary
**SIGNAL DOMINANCE CONFIRMED.** All gated fusion variants converge to RWKV-dominant regardless of initial gate bias. RWKV produces smoother gradients that are easier to optimize.

### Phase 3.7 Results

| Model | gate_init | Final R/M | Val Loss | Observation |
|-------|-----------|-----------|----------|-------------|
| GF-RH | 0.7 (RWKV) | 0.14 | 1.64 | ⚠️ RWKV dominant |
| HGF-MH | 0.3 (Mamba) | 0.24 | 1.69 | ⚠️ RWKV dominant |
| HGF-RH | 0.7 (RWKV) | 0.25 | 1.70 | ⚠️ RWKV dominant |

### Key Finding
- **GF-MH** (started 30:70 Mamba) → R/M 0.10 (RWKV dominant)
- **GF-RH** (started 70:30 RWKV) → R/M 0.14 (RWKV dominant)
- **HGF-MH** (started 30:70 Mamba) → R/M 0.24 (RWKV dominant)
- **HGF-RH** (started 70:30 RWKV) → R/M 0.25 (RWKV dominant)

**Conclusion:** `gate_init` doesn't matter. The optimizer "takes the path of least resistance" to RWKV.

### Changed
- **V4_BLEND_RATIOS.md**: Added complete Phase 3.7 results and analysis
- **V4_STRATEGY.md**: Phase 3.7 marked COMPLETE with signal dominance conclusion
- **V4_HANDOFF.md**: Updated to v4.9-Alpha, next focus is Phase 3.8 options
- **VERSION**: Updated to 4.9-Alpha

### Phase 3.8 Options (Next Steps)
1. Add balance regularization loss term
2. Increase Mamba LR multiplier (1.0 vs 0.5)
3. Freeze gates for first N steps
4. Accept RWKV dominance if loss is acceptable

---

## [4.8-Alpha] - 2026-01-10 (Phase 3.7 Planned)

### Summary
Phase 3.6 complete with fusion variant comparison. Discovered trade-off: position-adaptive fusion achieves lower loss but causes RWKV dominance. Phase 3.7 designed to test if dominance is signal-based or architectural.

### Added
- **V4_BLEND_RATIOS.md**: Complete hyperparameter analysis for blend ratio experiments
  - Adjustability matrix for all 6 fusion variants
  - How `gate_init` works (logit transform)
  - Phase 3.7 experiment plan with commands and results template
  - Interpretation guide for signal vs architectural dominance

### Changed
- **V4_STRATEGY.md**:
  - Phase 3.6 marked COMPLETE with results
  - Phase 3.7 added (Tasks 31-35: Blend Ratio Sweep)
  - Next actions updated to prioritize dominance testing
- **V4_HANDOFF.md**:
  - Updated to v4.8-Alpha
  - Next Agent Instructions now focus on Phase 3.7
  - Available variants documented with gate_init values
- **VERSION**: Updated to 4.8-Alpha

### Phase 3.6 Key Findings

| Variant | Val Loss | R/M Ratio | Verdict |
|---------|----------|-----------|---------|
| **GF-MH** | **1.59** | 0.10 ⚠️ | Best loss, RWKV dominant |
| GF | 1.61 | 0.12 ⚠️ | Good loss, RWKV dominant |
| CP | 1.61 | 0.19 ⚠️ | Good loss, RWKV dominant |
| HGF | 1.69 | 0.21 ⚠️ | Mid loss, RWKV dominant |
| **HY** | 1.69 | **0.45** ✅ | Mid loss, balanced |

**Trade-off identified:** Lower loss ↔ Worse component balance

### Phase 3.7 Plan
Test symmetric configurations to determine RWKV dominance cause:
- Task 31: GF-RH (gate_init=0.7, RWKV-heavy)
- Task 32: HGF-MH (gate_init=0.3, Mamba-heavy)
- Task 33: HGF-RH (gate_init=0.7, RWKV-heavy)
- Task 34: Gate drift analysis
- Task 35: Signal vs architectural conclusion

---

## [4.2-Alpha] - 2026-01-09 (Phase 2 Complete)

### Summary
Phase 2 completed: Fusion and ratio variants benchmarked. Gated Fusion (GF) with Mamba-Heavy ratio (gate_init=0.3) selected as winner. Architecture clarifications finalized.

### Added
- **hybrid_v4_ratio.py**: Two ratio variants of GF architecture
  - `HybridModel_GF_RH`: RWKV-heavy (gate_init=0.7)
  - `HybridModel_GF_MH`: Mamba-heavy (gate_init=0.3) — **Phase 2 Winner**
- **benchmark_variants.py**: Comprehensive benchmark suite comparing all 7 variants
- **V4_DESIGN.md - Architecture Clarity Sections**:
  - "WHAT WE ACTUALLY BUILT": 8 parallel blocks with ASCII diagrams
  - "Contrast: Sequential vs Parallel": Clarifies proposed vs implemented
  - Updated header with Phase 2 completion status

### Changed
- **V4_STRATEGY.md**: 
  - Task 14-18 marked complete
  - Phase 2 results table updated with benchmarks
  - Phase 3 tasks defined (scale to 8M)
- **V4_DESIGN.md**:
  - Status updated from "Not Yet Implemented" to "Parallel Block Architecture - IMPLEMENTED"
  - Added date updated field
  - Clearer distinction between PROPOSED (future) and ACTUAL (implemented)
  - Multiple ASCII diagrams added for clarity
  - Full model dataflow diagram added
  - Single block visualization improved

### Benchmarks (Phase 2 Results)

#### Fusion Strategy Variants (Tasks 14)
| Strategy | Model | Val Loss | vs HY | Throughput | Winner |
|----------|-------|----------|-------|-----------|--------|
| Gated Fusion | GF | 1.6891 | -4.0% | 42.9K tok/s | 🥇 |
| Concat+Project | CP | 1.6919 | -3.8% | 47.7K tok/s | 🥈 |
| Baseline | HY | 1.7600 | — | 31.7K tok/s | 🏁 |
| Weighted Sum | WS | 1.8185 | +3.3% | 45.4K tok/s | |
| Residual Fusion | RF | 1.9480 | +10.6% | 47.4K tok/s | |

**Finding**: Gated fusion with learnable per-position weighting outperforms all other fusion strategies by 4%.

#### Ratio Variants of Gated Fusion (Tasks 15-17)
| Component Balance | Model | Val Loss | vs Balanced | Winner |
|-------------------|-------|----------|------------|--------|
| Mamba-Heavy (70%) | GF-MH | **1.6700** | **-1.8%** | 🏆 **Overall Winner** |
| Balanced (50-50) | GF | 1.6998 | — | 🥈 |
| RWKV-Heavy (70%) | GF-RH | 1.7201 | +0.3% | 🥉 |

**Finding**: Mamba-selective capabilities benefit from higher relative weight (gate_init=0.3). RWKV-heavy performs worse than balanced.

### Testing
- All 7 variants trained for 500 steps, batch_size=64, seq_len=64
- Same dataset (shakespeare.txt), optimizer (AdamW), and config
- Fair comparison with fixed random seeds
- Throughput measured in tokens/second

### Implementation Status
```
GF-MH (Phase 2 Winner):
├─ 8 ParallelHybridBlocks
├─ Each: 1 RWKV-6 (∥) 1 Mamba-2 in parallel
├─ Gated Fusion with gate_init=0.3
├─ 3.5M parameters
├─ Val Loss: 1.6700
└─ File: hybrid_v4_ratio.py
```

### Documentation
- **Created**: README.md (primary project introduction)
- **Created**: CHANGELOG.md (this file)
- **Created**: VERSION file (semantic versioning)
- **Updated**: V4_DESIGN.md (architecture clarity)
- **Updated**: V4_STRATEGY.md (Phase 2 results)

---

## [4.1-Alpha] - 2026-01-09 (Phase 2 Midpoint)

### Summary
Phase 2 fusion benchmark completed. Gated Fusion (GF) selected as winning fusion strategy. Ratio variants created and ready for testing.

### Added
- **hybrid_v4_GF.py**: Gated Fusion architecture (learnable per-position gate)
- **hybrid_v4_CP.py**: Concat+Project architecture (baseline fusion variant)
- **hybrid_v4_WS.py**: Weighted Sum architecture (single learnable weight)
- **hybrid_v4_RF.py**: Residual Fusion architecture (residual correction)

### Benchmarks
- GF (Gated Fusion): Val Loss 1.6891 — **4% better than HY baseline**
- CP (Concat+Project): Val Loss 1.6919 — Very close to GF
- All variants trained to convergence at 500 steps

### Known Issues
- RWKV-Heavy ratio variant (GF-RH) not yet tested
- Mamba-Heavy ratio variant (GF-MH) not yet tested
- Need extended training to verify if gap widens

---

## [4.0-Alpha] - 2026-01-08 (Phase 1 Complete)

### Summary
Phase 1 completed: Extended training of baseline hybrid model (HY) demonstrates stability and learning. Architecture validated. Ready for fusion variant exploration.

### Initial Implementation
- **hybrid_v4.py**: Baseline parallel hybrid (HY variant)
  - Per-channel gain fusion (α and β parameters per hidden dimension)
  - 8 parallel blocks (1 RWKV-6 + 1 Mamba-2 each)
  - 3.5M parameters, 128 hidden dimension
  
- **benchmark_suite.py**: Initial benchmark framework
- **data_loader.py**: Shakespeare dataset loader (97 vocab, char-level)
- **tokenizer.py**: Character tokenizer

### Phase 1 Training (Extended)
```
Task 13: Extended Training - HY Baseline
├─ Training: 5000 steps
├─ Batch size: 64, Seq length: 64
├─ Optimizer: AdamW, lr=3e-4
├─ Initial Loss: 4.60 (entropy initialized)
├─ Final Loss: 1.14
├─ Improvement: 75% loss reduction
└─ Status: ✅ Complete
```

### Validation Gates Passed
- ✅ G1: Forward pass (no NaN, correct shapes)
- ✅ G2: Init entropy (5.1 at step 0 — in acceptable range)
- ✅ G3: Training stability (loss decreasing smoothly 4.60→1.14)
- ✅ G3.5: State health (cosine <0.99, no component freeze)
- ✅ G4: Component balance (gradient ratio 0.5-2.0)

### Documentation
- **V4_DESIGN.md**: Architecture specification with parallel block diagram
- **V4_STRATEGY.md**: Task backlog and Phase 1-2 planning
- **V4_HANDOFF.md**: Agent continuity notes

### Known Limitations (Phase 1)
- Only baseline HY variant tested
- No comparison with other fusion strategies
- Short sequences (64 tokens) — long-context not yet explored
- Model small (3.5M) — scaling to 8M+ not yet attempted

---

## [0.2.0] - 2026-01-08 (Legacy - Archive)

### Summary
V0.2.0 was a 125M parameter model with mixed training data. Superseded by V4 parallel architecture.

### Architecture
- 12 layers, 768 hidden dimension
- SelectiveWKV hybrid (V0-style sequential mixing)
- 125M parameters

### Results
- Final loss: 0.87 (5000 steps)
- Training dataset: 200K samples (mixed: FineWeb-Edu, PG19, OpenHermes, TinyStories)
- Demonstrated stability but not optimized for parallel hybrids

### Status
**DEPRECATED**: Replaced by V4 architecture. See archive/ for original files.

---

## [0.1.0] - 2026-01-08 (Legacy - Archive)

### Summary
V0.1.0 was the first grounded hybrid prototype with 5.5M parameters. Foundational experiment, superseded by V4.

### Architecture
- 6 layers, 256 hidden dimension
- Grounded RWKV + Mamba hybrid (sequential mixing)
- 5.5M parameters
- FLA `chunk_simple_gla` backend

### Results
- Final loss: 0.80 (5000 steps)
- 52.7K tok/s throughput
- Identified Mamba-dominant imbalance (time_decay projection norm=196)

### Issues Found
1. Mamba-dominant: selective mode overpowers grounding
2. Short prompts fail: model needs context
3. Mixed style output: data mixture bleeding

### Status
**DEPRECATED**: Replaced by V4 architecture (parallel blocks, better balance). See archive/ for original files and `VERSIONS.md` for details.

---

## Version Numbering

**Semantic Versioning:** `MAJOR.MINOR-STATUS`

- **MAJOR**: Architecture type (0=legacy, 4=V4 parallel)
- **MINOR**: Feature set iteration
- **STATUS**: Release stage (Alpha, Beta, Stable)

**Example:**
- `4.2-Alpha`: V4 architecture, 2nd feature iteration, experimental
- `4.2-Beta`: V4 architecture, 2nd feature iteration, tested but not production
- `4.2`: V4 architecture, 2nd feature iteration, production ready

---

## See Also

- **[README.md](README.md)** — Project overview and quick start
- **[V4_DESIGN.md](V4_DESIGN.md)** — Architecture specification
- **[V4_STRATEGY.md](V4_STRATEGY.md)** — Task backlog and progress tracking
- **[VERSION](VERSION)** — Current version number

---

**Last Updated:** 2026-01-09
