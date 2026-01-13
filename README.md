# GroundThink: Hybrid RWKV-6 + Mamba-2 Architecture

**Status:** V0.5.0.5 — Phase 0 Complete, Phase 1 Starting  
**V4 Status:** ✅ Graduated (GPT-2 parity at 17% fewer params)  
**Phase 0:** ✅ COMPLETE (Base model characterization)  
**Updated:** 2026-01-13  
**Repository:** https://github.com/9to5ninja-projects/groundthink  
**License:** MIT (see [LICENSE](LICENSE))

> ⚠️ **EXPERIMENTAL RESEARCH CODE** — Not for production use. No warranties.
>
> ⚖️ **ATTRIBUTION:** This project builds on RWKV-6 (Peng et al., 2024) and Mamba-2 (Dao & Gu, 2024). Our contribution is the fusion architecture, training methodology, and validation framework. See [ATTRIBUTION.md](ATTRIBUTION.md) for full citations.

---

## Quick Links

- 📖 [About GroundThink](ABOUT.md) — Project overview, status, and goals
- ⚖️ [Attribution & Citations](ATTRIBUTION.md) — **Required reading for usage/citation**
- 🚀 [Getting Started](GETTING_STARTED.md) — Installation and setup
- 🗺️ [Documentation Map](DOCUMENTATION_MAP.md) — Full documentation index
- 📊 [V4 Graduation Summary](#v4-graduation-summary) — Phase 4.0 results
- 🔬 [Phase 0 Findings](#phase-0-base-model-characterization) — **COMPLETE**
- 🔮 [V0.5 Roadmap](V0.5_ROADMAP.md) — Twin Debate architecture plan (Phase 1 current)

---

## What's New: Phase 0 Complete → Phase 1 Starting

**Phase 0 Complete ✅ (2026-01-13):**
- ✅ Pure RWKV-6 benchmarked (4M params) — **AMPLIFIER** (5.5x total variance)
- ✅ Pure Mamba-2 benchmarked (4M params) — **AMPLIFIER** at full model (2.0x), **DAMPER** at layer level
- ✅ GPT-1 baseline benchmarked (4M params) — **AMPLIFIER** (782x extreme)
- ✅ BlinkDL initialization confirmed architecture-agnostic (fixes saturation in all models)
- ✅ Comparative analysis complete — Fusion architecture decisions made

**Key Discovery:** All full models amplify variance, but SSMs are 142× more stable than attention-based models. RWKV amplifies per-layer, Mamba damps at layer level—complementary behavior confirmed!

**Phase 1 Now Starting:**
- Task 0.1: GRU Arbiter (stateful gating)
- Task 0.2: Mamba Residual Path (preserve damping)
- Task 0.3: Twin Debate Loss (pathway specialization)
- Task 0.4: 4M Pilot Run (target: Mamba >5% contribution)

See [V0.5_ROADMAP.md](V0.5_ROADMAP.md) and [BASE_MODEL_CHARACTERIZATION.md](BASE_MODEL_CHARACTERIZATION.md) for details.

---

## Phase 0: Base Model Characterization

### Summary Table

| Model | Type | Variance Amplification | Key Insight |
|-------|------|------------------------|-------------|
| GPT-1 (4M) | Attention | **782×** | Extreme amplification |
| RWKV-6 (4M) | SSM | **5.5×** (1.28×/layer) | Amplifies, layer-level |
| Mamba-2 (4M) | SSM | **2.0×** full / **0.005×** layer | Damps at layer level! |

### Architecture Decisions for Phase 1

1. **Layer-Level Fusion:** Preserve Mamba's damping by fusing before residual aggregation
2. **BlinkDL Init:** Apply to all components (embeddings: ±1e-4, projections: zero)
3. **Target Variance:** 2–6× total (SSM range, not GPT-1's 782×)
4. **Open Question:** How to add Mamba residuals without losing damping effect?

See [BASE_MODEL_CHARACTERIZATION.md](BASE_MODEL_CHARACTERIZATION.md) for full findings.

---

## What is GroundThink?

GroundThink is an **experimental hybrid architecture** combining:
- **RWKV-6** (Peng et al., 2024) — recurrent-style, long-range memory
- **Mamba-2** (Dao & Gu, 2024) — selective state-space model
- **Gated Fusion** (our contribution) — learnable pathway weighting

**Our Contribution:** The specific fusion mechanism, training methodology, and validation framework. We did not create RWKV-6 or Mamba-2 — we are exploring how to optimally combine them.

Both components run **in parallel within each block**, fused via learned gating. This design leverages RWKV's recurrent continuity and Mamba's selective reasoning in a single forward pass.

**Key innovation:** Learned α-gating enables context-dependent pathway weighting, allowing the model to dynamically choose between recurrent (RWKV) and selective (Mamba) processing modes.

---

## Architecture Overview

### The Building Block: ParallelHybridBlock

```
┌─────────────────────────────────────┐
│       Input: [batch, seq, 128]      │
├─────────────────────────────────────┤
│                                     │
│  Norm                               │
│  ├─→ RWKV-6 ──┐                    │
│  └─→ Mamba-2 ─┤                    │
│               ▼                     │
│         Gated Fusion (learns α)    │
│         output = α·rwkv + (1-α)·mamba
│               │                     │
│               + SKIP ────────────→  │
│               │                     │
│               ▼                     │
│         RMSNorm + FFN              │
│               │                     │
│               + SKIP ────────────→  │
│               │                     │
│               ▼                     │
│     Output: [batch, seq, 128]      │
│                                     │
└─────────────────────────────────────┘
```

**See [V4_DESIGN.md](V4_DESIGN.md) for detailed architecture diagrams and layer specifications.**

---

## V4 Graduation Summary

### Key Results (Phase 4.0 Complete)

| Metric | Result | Comparison |
|--------|--------|------------|
| **GPT-2 Parity** | Loss ratio 1.008 | ✅ EQUIVALENT |
| **Parameter Efficiency** | 5.6M params | 17% fewer than GPT-2 (6.8M) |
| **Dataset** | WikiText-103 | 16K BPE tokenization |
| **Long Context** | 1.04× @ 512 tokens | Stable degradation |
| **Throughput** | 42.9K tok/s | 4.5× slower (kernel optimization needed) |

### Critical Findings

**1. Mamba Paradox:**
- Mamba receives 10× larger gradients than RWKV
- But contributes <0.3% to final state
- Architectural behavior, not a training bug

**2. Attractor Zone:**
- All gate initializations converge to 10-30% RWKV/Mamba ratio
- Optimizer finds same equilibrium regardless of starting bias

**3. Architecture Validated:**
- Hybrid fusion matches transformer performance at small scale
- Linear O(n) complexity maintained for both pathways
- Ready for V0.5 architectural improvements

See [OBSERVATION_SYNTHESIS.md](OBSERVATION_SYNTHESIS.md) for detailed analysis.

---

## Quick Start

### Requirements

```bash
# Install dependencies (Python 3.10+, CUDA 12.1+)
pip install -r requirements.txt

# On Linux, install optional faster kernels
pip install causal-conv1d mamba-ssm
```

### Run Benchmarks

```bash
# 1. Setup
git clone https://github.com/9to5ninja-projects/groundthink.git
cd groundthink
source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify environment
python -m tests.test_phase0_complete

# 3. Run benchmark
python benchmark_variants.py
```

---

## Documentation Map

**Essential reading:**
1. **[ONBOARDING.md](ONBOARDING.md)** — What are RWKV and Mamba? Why combine them?
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** — Clone, install, run first benchmark
3. **[V0.5_ROADMAP.md](V0.5_ROADMAP.md)** — Current phase implementation plan
4. **[V4_DESIGN.md](V4_DESIGN.md)** — Architecture specification

**Current status:**
- **[HANDOFF.md](HANDOFF.md)** — Agent handoff, current tasks
- **[BASE_MODEL_CHARACTERIZATION.md](BASE_MODEL_CHARACTERIZATION.md)** — Phase 0 findings
- **[CHANGELOG.md](CHANGELOG.md)** — Version history

---

## Contributing

Contributions follow the **survival of the fittest** approach:

1. Create a new variant (fork hybrid_v4_GF.py)
2. Benchmark it against current winner (GF-MH)
3. If it beats the winner, merge it
4. Update README with new results

The only gate: **must benchmark fairly** (same dataset, same steps, same seeds).

---

## License

MIT (see [LICENSE](LICENSE))

---

## Questions?

See documentation in this order:
1. **Current Phase:** [V0.5_ROADMAP.md](V0.5_ROADMAP.md)
2. **Architecture:** [V4_DESIGN.md](V4_DESIGN.md)
3. **Status:** [HANDOFF.md](HANDOFF.md)
4. **Phase 0 Findings:** [BASE_MODEL_CHARACTERIZATION.md](BASE_MODEL_CHARACTERIZATION.md)

---

**Last Updated:** 2026-01-13 (Phase 0 Complete, Phase 1 Starting)
