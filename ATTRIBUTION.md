# GroundThink Attribution & Intellectual Property

**Project**: GroundThink — Dual-Pathway Linear-Complexity Language Models  
**Lead Researcher**: 9to5ninja  
**Date**: January 2026

---

## Core Components (Prior Work — NOT Ours)

### RWKV-6 Architecture
- **Paper**: "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"
- **Authors**: Bo Peng et al.
- **Year**: 2024
- **Source**: https://github.com/BlinkDL/RWKV-LM
- **License**: Apache 2.0
- **What we use**: RWKV-6 attention mechanism via FLA (Flash Linear Attention) library
- **How we use it**: One of two parallel pathways in our hybrid architecture

**Citation**:
```bibtex
@article{peng2024eagle,
    title={Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence},
    author={Peng, Bo and others},
    year={2024}
}
```

### Mamba-2 Architecture
- **Paper**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- **Authors**: Tri Dao, Albert Gu (Princeton/CMU)
- **Year**: 2024
- **Source**: https://github.com/state-spaces/mamba
- **What we use**: Mamba-2 selective SSM architecture via FLA library
- **How we use it**: Second of two parallel pathways in our hybrid architecture

**Citation**:
```bibtex
@article{dao2024mamba,
    title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
    author={Dao, Tri and Gu, Albert},
    year={2024}
}
```

### Structured State Space Models (S4)
- **Paper**: "Efficiently Modeling Long Sequences with Structured State Spaces"
- **Authors**: Albert Gu, Karan Goel, Christopher Ré
- **Year**: 2021
- **Foundation**: Both RWKV-6 and Mamba-2 build on S4 concepts

**Citation**:
```bibtex
@inproceedings{gu2021efficiently,
    title={Efficiently Modeling Long Sequences with Structured State Spaces},
    author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
    booktitle={ICLR},
    year={2022}
}
```

### NVIDIA Hybrid Architecture Study
- **Source**: NVIDIA Research on Hybrid SSM Architectures
- **Finding**: ~43% Mamba-2 + attention outperforms pure models
- **Year**: 2024
- **How we use it**: Validates hybrid approach, informs our target fusion ratios

### VADER Sentiment Analysis
- **Paper**: "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text"
- **Authors**: C.J. Hutto, Eric Gilbert
- **Year**: 2014
- **What we use**: Sentiment polarity/intensity scoring
- **Future use**: Environmental sensor for semantic weighting (not yet implemented)

**Citation**:
```bibtex
@inproceedings{hutto2014vader,
    title={VADER: A Parsimonious Rule-based Model for Sentiment Analysis},
    author={Hutto, C.J. and Gilbert, Eric},
    booktitle={ICWSM},
    year={2014}
}
```

### NRC Emotion Lexicon
- **Paper**: "Crowdsourcing a Word-Emotion Association Lexicon"
- **Authors**: Saif Mohammad, Peter Turney
- **Year**: 2013
- **What we use**: Emotion intensity scores (anger, fear, joy, etc.)
- **Future use**: Environmental sensor for semantic weighting (not yet implemented)

**Citation**:
```bibtex
@article{mohammad2013crowdsourcing,
    title={Crowdsourcing a Word-Emotion Association Lexicon},
    author={Mohammad, Saif M and Turney, Peter D},
    journal={Computational Intelligence},
    year={2013}
}
```

### spaCy NLP Library
- **Source**: https://spacy.io/
- **Authors**: Explosion AI
- **What we use**: Entity recognition, dependency parsing
- **Future use**: Syntactic analysis for semantic weighting (not yet implemented)

### Flash Linear Attention (FLA) Library
- **Source**: https://github.com/sustcsonglin/flash-linear-attention
- **Authors**: Songlin Yang et al.
- **What we use**: Efficient CUDA implementations of RWKV-6 and Mamba-2
- **Critical**: Our entire implementation depends on FLA

**Citation**:
```bibtex
@misc{yang2024fla,
    title={Flash Linear Attention},
    author={Yang, Songlin and others},
    year={2024}
}
```

### Mixture of Experts (MoE) Concept
- **Paper**: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- **Authors**: Noam Shazeer et al.
- **Year**: 2017
- **Inspiration**: Gating mechanisms and expert routing influenced our fusion design

**Citation**:
```bibtex
@article{shazeer2017outrageously,
    title={Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
    author={Shazeer, Noam and others},
    journal={ICLR},
    year={2017}
}
```

### WikiText-103 Dataset
- **Paper**: "Pointer Sentinel Mixture Models"
- **Authors**: Stephen Merity et al.
- **Year**: 2016
- **What we use**: Training and evaluation dataset

**Citation**:
```bibtex
@article{merity2016pointer,
    title={Pointer Sentinel Mixture Models},
    author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
    journal={arXiv preprint arXiv:1609.07843},
    year={2016}
}
```

---

## Novel Contributions (OURS)

### 1. Gated RWKV-6 + Mamba-2 Fusion Architecture
**What existed before:**
- RWKV-6 (standalone)
- Mamba-2 (standalone)
- Hybrid transformers (RWKV+attention, Mamba+attention)

**What we created:**
- First published combination of RWKV-6 + Mamba-2 specifically
- Context-dependent α-gating mechanism (learned weighting)
- Parallel pathway architecture with residual connections
- "Pathways-as-twins" philosophical framing

**Implementation**: `models/hybrid_v4.py` (ParallelHybridBlock)

### 2. Multi-Component Loss Function
**What existed before:**
- MoE router loss (load balancing)
- Multi-task learning losses
- Twin network training

**What we created:**
- **L_diversity**: Cosine similarity penalty encouraging pathway disagreement
- **L_arbiter**: Reward/penalty for correct gating decisions
- **L_mode**: Enforce SL (RWKV-heavy) vs. LS (Mamba-heavy) compliance
- **L_spatial**: Spatial consistency across INPUT/THINK/OUTPUT segments
- Staged loss activation schedule (weeks 2-4)
- Specific loss weights and formulations

**Implementation**: `tools/debate_loss.py` (planned)

### 3. Qualia Preservation System
**What existed before:**
- Curriculum learning
- Knowledge distillation
- Gradual unfreezing

**What we created:**
- Three-phase fade schedule (guided → internalization → autonomy)
- Control embeddings as "mood lighting" (semantic → 8-dim compression)
- Loss scaling with fade-out (1.0 → 0.1)
- Explicit "implicit learning" methodology

**Implementation**: `train.py` (planned enhancement)

### 4. Systematic Validation Methodology
**What existed before:**
- Standard ablation studies
- Benchmark comparisons
- Toy task testing

**What we created:**
- **Pathway specialization testing** via gradient flow analysis
- **Fusion effectiveness** via gate histogram analysis
- **Synergy measurement** via compositional tasks
- **State dynamics characterization** (S0-S4 tests)
- 4-week validation protocol with specific gates
- Base model characterization before fusion (Phase 0)

**Implementation**: `tests/` directory (S0-S4 tests, NIAH, ablation, etc.)

### 5. Base Model Characterization Methodology (Phase 0)
**What existed before:**
- Standard baseline comparisons
- Component ablation studies

**What we created:**
- Systematic characterization of RWKV-6, Mamba-2, GPT-1 separately before fusion
- Variance analysis to determine stabilizer vs. destabilizer roles
- NIAH testing adapted for BPE tokenization
- Evidence-based fusion design (characterize before combining)

**Implementation**: `BASE_MODEL_CHARACTERIZATION.md`, `models/rwkv6_pure.py`, `tools/variance_analysis.py`

### 6. Semantic Weighting as Environmental Sensors
**What existed before:**
- VADER/NRC/spaCy as text filters
- Sentiment-based data selection

**What we created (planned):**
- Environmental sensors (not filters) - all data goes through
- RWKV/Mamba bias based on linguistic properties (sentiment, syntax, complexity)
- "Mood lighting" metaphor for architectural influence
- Integration into gating mechanism

**Implementation**: Planned for Phase 1

---

## Licensing & Usage

### Our Code
- **License**: Apache 2.0 (to be compatible with RWKV/Mamba)
- **Copyright**: © 2026 9to5ninja
- **Requirement**: Must cite both our work AND all prior work listed above

### If You Use This Code
Please cite:
```bibtex
@misc{groundthink2026,
    title={GroundThink: Dual-Pathway Linear-Complexity Language Models 
           with Learned Arbitration},
    author={9to5ninja},
    year={2026},
    url={https://github.com/9to5ninja-projects/groundthink}
}
```

**AND cite all applicable prior works** (especially RWKV-6, Mamba-2, FLA).

---

## What We're Building On vs. What We're Creating

### Simple Analogy: Building a House

**We didn't invent:**
- Bricks (RWKV-6)
- Steel beams (Mamba-2)
- Cement (VADER/NRC/spaCy)
- Electrical wiring (Transformer concepts)

**We DID invent:**
- The specific floor plan (how RWKV+Mamba connect)
- The structural design (gating mechanism)
- The climate control system (qualia preservation)
- The construction methodology (validation protocol)

We don't claim we invented bricks.  
We DO claim we invented this specific house design.  
**That's valid. That's research.**

---

## Research Ethics Statement

This project follows standard research ethics:

1. **Attribution**: All prior work is properly cited
2. **Transparency**: Clear distinction between borrowed and novel components
3. **Respect**: We acknowledge the researchers whose work we build on
4. **Honesty**: We do not claim to have invented foundational components

> "If I have seen further, it is by standing on the shoulders of giants."  
> — Isaac Newton

Every ML paper builds on prior work. The value is in the specific combination, methodology, and findings.

---

## Status: January 2026

- **V4 Achievement**: GPT-2 parity (loss ratio 1.008) with 17% fewer parameters
- **Current Phase**: Phase 0 (Base Model Characterization) before hybrid fusion
- **Architecture Status**: Pure RWKV-6 implementation ready for baseline testing
- **Validation Status**: Diagnostic tooling complete, training pending

---

**Last Updated**: 2026-01-11  
**Maintainer**: 9to5ninja  
**Contact**: https://github.com/9to5ninja-projects
