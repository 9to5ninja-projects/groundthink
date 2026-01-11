# About GroundThink

**GroundThink** is an experimental hybrid language model architecture combining RWKV-6 and Mamba-2 pathways with learnable gated fusion.

**Attribution:** This project builds on RWKV-6 (Peng et al., 2024) and Mamba-2 (Dao & Gu, 2024) architectures. Our contribution is the specific fusion mechanism, training methodology, and validation framework. See [ATTRIBUTION.md](ATTRIBUTION.md) for full citations.

---

## What Is This Project?

GroundThink explores whether **dual-pathway linear-complexity models** can match transformer performance while maintaining O(n) efficiency. The architecture runs RWKV-6 (recurrent-style) and Mamba-2 (selective state-space) in parallel, fusing their outputs via learned gating.

**Key Innovation (Our Contribution):** Context-dependent pathway weighting allows the model to dynamically choose between recurrent continuity (RWKV) and selective reasoning (Mamba) based on input characteristics.

**Building On:** RWKV-6 architecture (Peng et al., 2024), Mamba-2 architecture (Dao & Gu, 2024), structured state space models (Gu et al., 2021).

---

## Project Status

| Phase | Status | Key Results |
|-------|--------|-------------|
| **V4 (Phase 4.0)** | ✅ **Graduated** | GPT-2 parity at 17% fewer params |
| **V0.5 Phase 0** | 🔄 **Active** | Characterizing pure RWKV-6 & Mamba-2 |
| **V0.5 Phase 1** | ⏸️ **Pending** | Fusion design (informed by Phase 0) |

### V4 Achievements (Completed January 2026)
- **GPT-2 Equivalent Performance:** Loss ratio 1.008 on WikiText-103
- **Parameter Efficiency:** 5.6M params vs GPT-2's 6.8M (17% reduction)
- **Linear Complexity:** Maintained O(n) time for both pathways
- **Long Context Stability:** 1.04x degradation at 512 tokens vs 64 tokens

### V4 Key Findings
1. **Mamba Paradox:** Mamba receives 10× larger gradients but contributes <0.3% to state
2. **Attractor Zone:** All gate initializations converge to 10-30% RWKV/Mamba ratio
3. **Architecture Validated:** Hybrid fusion competitive with transformers at small scale

### V0.5 Phase 0 Goals (Current Focus)
- **Pure RWKV-6 Benchmark:** Characterize recurrent pathway behavior at 4M scale
- **Pure Mamba-2 Benchmark:** Characterize selective state-space behavior at 4M scale
- **GPT-1 Baseline:** Fair comparison at matched parameter count
- **Comparative Analysis:** Document strengths/weaknesses to inform fusion design

### V0.5 Phase 1 Goals (After Phase 0)
- **Informed Fusion Design:** Use Phase 0 findings to design optimal architecture
- **Stateful Arbiter:** GRU vs stateless gating decision based on characterization
- **Twin Debate Loss:** If specialization needed per Phase 0 analysis
- **Implementation:** Execute fusion architecture with confidence

See [BASE_MODEL_CHARACTERIZATION.md](BASE_MODEL_CHARACTERIZATION.md) for Phase 0 plan and [V0.5_ROADMAP.md](V0.5_ROADMAP.md) for Phase 1.

---

## Research Goals

### Primary Questions
1. **Can linear-complexity hybrids match transformer performance?**
   - V4 answer: Yes, at small scale (5-8M params)
   - Phase 0: Characterize individual RWKV-6 and Mamba-2 behavior
   - Phase 1: Design informed fusion based on characterization

2. **Do dual pathways naturally specialize?**
   - V4 answer: Not without explicit regularization (RWKV dominates)
   - Phase 0: Understand why RWKV dominated in naive fusion
   - Phase 1: Use findings to design better specialization approach

3. **Is learned gating better than fixed mixing?**
   - V4 answer: Yes (gated fusion beat all fixed strategies)
   - Phase 0: Benchmark base models before complex gating
   - Phase 1: Design gating informed by pathway characteristics

### Why This Matters
- **Efficiency:** Linear complexity enables longer contexts at constant memory
- **Interpretability:** Pathway contributions can be measured and controlled
- **Architectural Research:** Tests whether "twin debate" improves over simple fusion

---

## Who Should Use This?

### ✅ Good Fit For:
- **ML Researchers** exploring alternatives to quadratic attention
- **Students** studying hybrid architectures and pathway fusion
- **Contributors** interested in RWKV-6 / Mamba-2 research
- **Experimenters** testing linear-complexity language models

### ❌ Not Suitable For:
- **Production deployments** (experimental code, not optimized)
- **General-purpose LLM serving** (small scale, research focus)
- **Commercial applications** (no warranties, see LICENSE)
- **Out-of-the-box usage** (requires ML/PyTorch expertise)

---

## Technical Foundation

### Architecture: Parallel Hybrid Blocks
**Our Contribution:**
```
Input → [RWKV-6 ∥ Mamba-2] → Gated Fusion → FFN → Output
```

**Building On:**
- RWKV-6: "Eagle and Finch: RWKV with Matrix-Valued States" (Peng et al., 2024)
- Mamba-2: "Mamba: Linear-Time Sequence Modeling" (Dao & Gu, 2024)
- Gating inspired by: Mixture-of-Experts (Shazeer et al., 2017)

Each block:
- **RWKV-6 Pathway:** Linear recurrence, smooth long-range dependencies
- **Mamba-2 Pathway:** Selective state-space, input-dependent gating
- **Fusion Gate (α):** Learns per-position weighting: `output = α·RWKV + (1-α)·Mamba`
- **FFN:** Standard feed-forward network with residual connections

### Research Lineage
- **RWKV-6 (Finch):** Matrix-valued states, dynamic recurrence (Peng et al., 2024)
- **Mamba-2:** State Space Duality (SSD), selective updates (Dao & Gu, 2024)
- **Gated Fusion:** Inspired by mixture-of-experts but with pathway specialization

See [groundthink_architecture_research.md](groundthink_architecture_research.md) for full technical documentation.

---

## Development Status

### Code Maturity
- ✅ **Core Architecture:** Stable, validated in V4
- ✅ **Training Pipeline:** Working, needs optimization
- ⚠️ **CUDA Kernels:** Functional but 4.5× slower than GPT-2
- ⬜ **V0.5 Features:** In planning phase (GRU arbiter, debate loss)

### Performance Characteristics
| Metric | V4 Performance | Notes |
|--------|----------------|-------|
| Training Speed | 42.9K tok/s | 4.5× slower than GPT-2 (kernel optimization needed) |
| Memory Usage | 534MB (5.6M) | 11% more than GPT-2 (dual pathways) |
| Validation Loss | 6.850 (WikiText) | Matches GPT-2 (6.798) |
| Long Context | 1.04× @ 512 tok | Stable, slight degradation |

---

## Contributing

**Current Status:** V0.5 planning phase — limited external contributions until architecture stabilizes.

**If you want to contribute:**
1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for documentation standards
2. Review [V0.5_ROADMAP.md](V0.5_ROADMAP.md) for upcoming work
3. Check [V4_HANDOFF.md](V4_HANDOFF.md) for current development context
4. Start with [GETTING_STARTED.md](GETTING_STARTED.md) to set up environment

**Best ways to help:**
- Test the architecture on different datasets
- Optimize CUDA kernels (current bottleneck)
- Validate findings on different hardware
- Suggest improvements to debate loss design

See [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md) for full documentation hierarchy.

---

## Citation

If you use GroundThink in research, please cite:

```bibtex
@software{groundthink2026,
  title={GroundThink: Hybrid RWKV-6 + Mamba-2 Architecture with Gated Fusion},
  author={9to5ninja},
  year={2026},
  url={https://github.com/9to5ninja/groundthink},
  note={V4 (Phase 4.0) — Experimental research code}
}
```

**AND please cite the foundational works:**
```bibtex
@article{peng2024eagle,
    title={Eagle and Finch: RWKV with Matrix-Valued States},
    author={Peng, Bo and others},
    year={2024}
}

@article{dao2024mamba,
    title={Mamba: Linear-Time Sequence Modeling},
    author={Dao, Tri and Gu, Albert},
    year={2024}
}
```

See [ATTRIBUTION.md](ATTRIBUTION.md) for complete citation requirements.

*(Update with proper publication information if/when published)*

---

## License

MIT License — See [LICENSE](LICENSE) for full text.

**Key Points:**
- ✅ Free to use, modify, distribute
- ⚠️ **No warranties** — experimental research code
- ⚠️ **Not validated** for production use
- ✅ Attribution appreciated (see Citation above)

---

## Acknowledgments

### Research Foundations
- **RWKV-6:** Created by Bo Peng et al. (2024) — "Eagle and Finch: RWKV with Matrix-Valued States"
- **Mamba-2:** Created by Tri Dao & Albert Gu (2024) — "Mamba: Linear-Time Sequence Modeling"
- **S4:** Structured State Spaces (Gu et al., 2021) — foundation for both RWKV and Mamba
- **FLA Library:** Flash Linear Attention (Yang et al., 2024) — efficient implementations we rely on

### Dependencies
- PyTorch, FLA (flash-linear-attention), mamba-ssm, causal-conv1d, triton
- WikiText-103 dataset (Merity et al., 2016)
- See [requirements.txt](requirements.txt) and [ATTRIBUTION.md](ATTRIBUTION.md) for full details

---

## Contact & Links

- **Repository:** https://github.com/9to5ninja/groundthink
- **Documentation:** [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md)
- **Issues:** (GitHub Issues if repository is public)

**Questions?** Check [GETTING_STARTED.md](GETTING_STARTED.md) first, then [V4_HANDOFF.md](V4_HANDOFF.md) for current status.
