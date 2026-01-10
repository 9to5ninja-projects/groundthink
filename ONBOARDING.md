# GroundThink V4 Onboarding Guide

**For:** New team members, students, external researchers, future agents  
**Purpose:** Understand the "why" and "big picture" before diving into implementation  
**Created:** 2026-01-10  
**Updated:** See CHANGELOG.md for project evolution

---

## Quick Start: What Are We Building?

**Problem:** Large language models use "Transformer Attention," which is powerful but **slow** at long sequences (O(n²) complexity). Can we build a hybrid that combines fast, lightweight models without sacrificing capability?

**Hypothesis:** Combining RWKV-6 (recurrent memory) and Mamba-2 (selective memory) in parallel can outperform each individually.

**Current Status:** ✅ Phase 2 Complete. Parallel hybrid with Gated Fusion selected as winner (GF-MH variant: 1.67 val loss vs HY baseline 1.76). Scaling to 8M parameters next.

**If this works:** Efficient LLM alternative for long-context applications, edge devices, and latency-critical systems.

---

## Part 1: The Architectures (Conceptual Foundation)

### Why This Matters
Standard Transformers compute attention like this:
```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V    # O(n²) — expensive!
```

For a 32K token context, this creates a 32K×32K matrix (1 billion operations per token).

**Linear-time sequence models** reformulate this to avoid the matrix, reducing complexity to O(n).

---

### RWKV-6: "RNN-Like Transformer"

**Core Idea:** Rewrite attention as a **recurrent linear transformation** instead of all-pairs comparison.

**Mental Model:**
```
Standard Transformer: "Compare current token to ALL previous tokens"
RWKV-6:              "Update a running state with decay"
```

**How it Works (Simplified):**

1. **Time Mixing** (the attention replacement):
   - Maintains a running sum weighted by exponential decay
   - Each token "remembers" the past but with fading importance
   - Like an RNN but with efficient matrix operations

2. **Channel Mixing** (standard FFN):
   - Per-token nonlinearity
   - Same as in any Transformer

**Key Parameters:**
- `time_decay`: Controls how fast memory fades (larger = longer memory)
- `time_first`: Initial decay rate
- Gates for gating outputs per head

**Why Use RWKV-6:**
- ✅ **Smooth memory decay** — Long-range dependencies with natural fading
- ✅ **Linear complexity** — Can handle very long sequences efficiently
- ✅ **Efficient inference** — No matrix multiplications; works as RNN
- ❌ **Weakness:** Can't "selectively forget" — treats all history the same

**Parameter Cost:** ~700K params per layer at 128 hidden dim (expensive!)

**Reference:** Official RWKV-LM repo, paper "RWKV: Reinventing RNNs for the Transformer Era"

---

### Mamba-2: "Selective State Spaces"

**Core Idea:** Build a **state-space model** that chooses what to remember based on input content.

**Mental Model:**
```
Standard Attention:  "Compare current token to ALL previous tokens"
Mamba-2:            "Choose which parts of history are relevant, then update state"
```

**How it Works (Simplified):**

1. **Input Projection** (expand hidden dim):
   - Linear layer expands representation

2. **Selective State Space** (the selectivity):
   - Maintains hidden state vector
   - **Input-dependent transitions** — what you remember depends on input
   - This is the magic: unlike RWKV's fixed decay, Mamba learns what's important

3. **Output Projection** (compress back to hidden dim):
   - Linear layer compresses output

**Key Parameters:**
- `expand`: How much to expand internal state (e.g., 2× or 4×)
- `A, B, C, D`: Matrices controlling state transitions (learned during training)
- `A_log`: Log of state transition matrix (numerical stability)

**Why Use Mamba-2:**
- ✅ **Content-aware memory** — Remembers important info, forgets irrelevant
- ✅ **Linear complexity** — Efficient even at very long contexts
- ✅ **Hardware efficient** — Fused kernels on modern GPUs
- ❌ **Weakness:** More abstract, harder to reason about than RWKV

**Parameter Cost:** ~67K params per layer at 128 hidden dim (10× smaller than RWKV!)

**Reference:** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Goel, 2024)

---

## Part 2: Why Combine Them? (The Hypothesis)

### The Trade-off
| Aspect | RWKV-6 | Mamba-2 |
|--------|--------|---------|
| Memory Style | Smooth decay | Content-selective |
| Complexity | O(n) | O(n) |
| Parameter Efficiency | Low (~700K/layer) | High (~67K/layer) |
| Long-Context | Good (natural decay) | Better (selective) |
| Fast Inference | Good (RNN-like) | Good (SSM) |
| Interpretability | Moderate | Low |

### The Insight
- **RWKV's strength:** Smooth, predictable memory with established patterns
- **Mamba's strength:** Selective, content-aware memory for sudden changes
- **Together:** A system that can smoothly interpolate between "remember everything" and "forget everything"

**Real-world analogy:**
- RWKV = GPS tracking a journey (continuous history)
- Mamba = Highlighting key landmarks (selective important points)
- Hybrid = Both (you know where you've been AND what mattered most)

### The GroundThink Approach: Parallel Blocks

Instead of sequencing them (RWKV → Mamba → Mamba → ... → RWKV), GroundThink runs both **in parallel** within each block:

```
Input
  ├──→ RWKV-6   ──┐
  └──→ Mamba-2  ──┼──→ Fusion (blend them) ──→ FFN ──→ Output
```

**Why parallel?**
1. **Gradient independence:** RWKV gradients don't compete with Mamba gradients
2. **Parameter isolation:** Easy to monitor each component separately
3. **Balanced computation:** Both get same computational budget

---

## Part 3: Implementation (How We Actually Built It)

### The Fusion Mechanism: Gated Fusion (GF)

Phase 2 tested 5 fusion strategies. **Gated Fusion won** because:

```
output = gate(x) * rwkv_output + (1 - gate(x)) * mamba_output
```

**Why this works:**
- Per-position blending: different tokens can trust different components
- Learned weighting: model decides what matters when
- Lightweight: only O(hidden_dim) additional parameters

**Phase 2 Results:**

| Fusion | Val Loss | vs Baseline |
|--------|----------|------------|
| Gated (GF) | 1.6891 | -4.0% ✅ |
| Concat+Project (CP) | 1.6919 | -3.8% |
| Weighted Sum (WS) | 1.8185 | +3.3% |
| Residual (RF) | 1.9480 | +10.6% |

**Winner:** Gated Fusion

### The Ratio Strategy: Mamba-Heavy (MH)

Phase 2 also tested **how much RWKV vs Mamba** the model should use.

Insight: Mamba's 10× parameter efficiency means you can have more Mamba layers for the same budget. What if you bias the gate to favor Mamba?

```
gate_init = 0.3  # Initialize gate to 70% Mamba, 30% RWKV
```

**Phase 2 Ratio Results:**

| Ratio | Val Loss | Insight |
|-------|----------|---------|
| Mamba-Heavy (70%) | 1.6700 | ✅ BEST |
| Balanced (50%) | 1.6998 | Baseline |
| RWKV-Heavy (70%) | 1.7201 | Worse |

**Winner:** Mamba-Heavy (GF-MH)

**Interpretation:** The model learns to rely more on Mamba's selectivity than RWKV's smoothness. This suggests selective memory is more important for the task at this scale.

---

## Part 4: The Current Project State

### Phase Progression

```
Phase 1: Validate Baseline (✅ Complete)
  → Build HY model (simple fusion)
  → Train 5000 steps
  → Achieve 4.60 → 1.14 loss (-75%)

Phase 2: Find Best Fusion (✅ Complete)
  → Test 5 fusion strategies
  → Test 3 ratio variants
  → Select GF-MH as winner (1.67 val loss)

Phase 3: Scale & Verify (⬜ Next)
  → Scale GF-MH to 8M parameters
  → Train 50K steps to convergence
  → Test long-context (NIAH benchmark)

Phase 4: Advanced Evaluation (⬜ Future)
  → LongBench evaluation
  → InfiniteBench (100K+ token contexts)
  → Optimization for production
```

### Key Files & What They Do

| File | Purpose | Key Insight |
|------|---------|------------|
| [V4_DESIGN.md](V4_DESIGN.md) | Architecture specification | Shows actual vs proposed designs, layer math |
| [V4_STRATEGY.md](V4_STRATEGY.md) | Task backlog & complexity | Ordered task list with assessment matrix |
| [V4_HANDOFF.md](V4_HANDOFF.md) | Agent continuity | Current status, git commits, quick audit checklist |
| [hybrid_v4_ratio.py](hybrid_v4_ratio.py) | Phase 2 winner code | GF-MH implementation (3.5M params) |
| [benchmark_variants.py](benchmark_variants.py) | Comparative testing | Benchmark suite for fair variant comparison |
| [README.md](README.md) | Quick start guide | How to run benchmarks, interpret results |

---

## Part 5: Understanding the Code (What to Read First)

### If You Want to...

**Understand the architecture:**
1. Read [V4_DESIGN.md](V4_DESIGN.md) Section "WHAT WE ACTUALLY BUILT" (ASCII diagrams)
2. Look at [hybrid_v4_ratio.py](hybrid_v4_ratio.py) `ParallelHybridBlock_GF` class
3. See how RWKV and Mamba are called (notice the tuple unpacking for RWKV)

**Run benchmarks:**
1. Ensure dependencies: `pip install -r requirements.txt`
2. Run: `python benchmark_variants.py`
3. View results in console (loss curves, throughput table)

**Train your own model:**
1. Study [train_v4.py](train_v4.py) or [train.py](train.py)
2. Modify config in your script
3. Run training with monitoring
4. See [V4_TRAINING_GUIDE.md](V4_TRAINING_GUIDE.md) for hyperparameter tuning

**Understand why GF-MH won:**
1. Look at Phase 2 benchmark results in [CHANGELOG.md](CHANGELOG.md)
2. Read [V4_STRATEGY.md](V4_STRATEGY.md) Section "Phase 2" for analysis
3. Compare logits/activations from different fusion strategies

---

## Part 6: Glossary of Terms

**Key Abbreviations:**

| Term | Meaning | Context |
|------|---------|---------|
| GF | Gated Fusion | Learned per-position blending of RWKV & Mamba |
| GF-MH | Gated Fusion + Mamba-Heavy | Phase 2 winner (gate_init=0.3) |
| GF-RH | Gated Fusion + RWKV-Heavy | Variant tested in Phase 2 (gate_init=0.7) |
| CP | Concat+Project | Fusion: concatenate + linear (tested, not winner) |
| WS | Weighted Sum | Fusion: single learnable weight (tested, not winner) |
| RF | Residual Fusion | Fusion: residual correction (tested, worst) |
| HY | Hybrid baseline | Original per-channel fusion (Phase 1) |
| GF | Gated Fusion | Learned per-position weighting |
| NIAH | Needle-in-a-Haystack | Long-context benchmark (find target in huge context) |
| LongBench | Long-context benchmark suite | Multiple task types at 4K-100K tokens |
| InfiniteBench | Ultra-long-context benchmark | Tasks up to 1M tokens |

**Technical Terms:**

| Term | Meaning | Why It Matters |
|------|---------|---------------|
| `time_decay` (RWKV) | Exponential decay rate for memory | Higher = longer memory, lower = shorter memory |
| `A_log` (Mamba) | Log of state transition matrix | Controls how selective memory update is |
| `gate_init` | Initial value for fusion gate | If 0.3, starts favoring Mamba (70%-30%) |
| `expand` (Mamba) | Internal state expansion ratio | 2×, 4× etc; larger = more expressive but slower |
| `head_dim` (RWKV) | Per-head dimension | Must divide hidden_size nicely |
| Validation Loss | Loss on held-out data | How well model generalizes |
| Throughput (tok/s) | Tokens per second | Speed metric; higher is faster |
| PPL (Perplexity) | e^(loss) | Intuitive loss metric; lower is better |

---

## Part 7: Mental Models for Decision-Making

### When Reading Code: Ask These Questions

**"Why does this variant exist?"**
- Look at Phase 2 results table
- Understand what failure mode it's trying to fix
- See if it helped or hurt

**"Why did we choose this parameter?"**
- Check V4_DESIGN.md "Training Configuration" section
- Look for cited sources (e.g., "V3 Cross-Ref 1.6")
- Check if it's been validated (gates G1-G4)

**"Why use Mamba here, RWKV there?"**
- Remember: RWKV costs 10× more params
- So: RWKV for high-impact decisions, Mamba for breadth
- Phase 2 showed Mamba-heavy works better—revise if you disagree

**"Is this the final architecture?"**
- No. Phase 3 will test 8M scale (currently 3.5M)
- Phases 4-5 will test long-context
- Architecture might change based on results

---

## Part 8: Where to Learn More

### External Resources (The "Why" of Linear Models)

| Topic | Resource | Why |
|-------|----------|-----|
| **RWKV Fundamentals** | [RWKV-LM GitHub](https://github.com/BlinkDL/RWKV-LM) | Official repo, papers, explanations |
| **RWKV Papers** | "RWKV: Reinventing RNNs for the Transformer Era" | Understand the math |
| **Mamba Fundamentals** | [Mamba GitHub](https://github.com/state-spaces/mamba) | Official repo with paper link |
| **Mamba Paper** | "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Goel, 2024) | Very readable; explains selectivity |
| **Linear-Time Models Survey** | Search for "Linear Attention" or "Efficient Transformers" | Understand the broader context |
| **Transformer Attention Math** | Jay Alammar's "Illustrated Transformer" | Understand what we're replacing |

### Internal Resources (The "How" of GroundThink)

| Document | Read When | Why |
|----------|-----------|-----|
| [V4_DESIGN.md](V4_DESIGN.md) | First (understanding) | Architecture spec with code examples |
| [README.md](README.md) | First (practical) | Quick start, how to run benchmarks |
| [CHANGELOG.md](CHANGELOG.md) | When confused about history | What changed and when |
| [V4_STRATEGY.md](V4_STRATEGY.md) | Before starting work | Task list, complexity assessment |
| [V4_HANDOFF.md](V4_HANDOFF.md) | Before any major work | Current state, audit checklist |
| [V4_BUILD_LOG.md](V4_BUILD_LOG.md) | When debugging specific issues | What was tried, what failed/worked |
| [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) | When optimizing performance | Profiling, benchmarking methodology |

---

## Part 9: Quick Mental Model (5-Minute Version)

**If you only read this section:**

### The Problem
Transformers are slow (O(n²)) for long sequences. Can we build something faster?

### The Solution
RWKV-6 + Mamba-2 in parallel:
- **RWKV:** "Smooth, exponential memory decay" (like an RNN)
- **Mamba:** "Smart, selective memory" (learns what to remember)
- **Together:** "Blend of both, per-position"

### The Result (Phase 2)
- Gated Fusion beats all other fusion strategies (-4% loss)
- Mamba-Heavy gate (70% Mamba) beats balanced (-1.8% loss)
- Current model: 3.5M params, 1.67 val loss

### Next Steps (Phase 3)
- Scale to 8M params
- Train 50K steps
- Test long-context (NIAH, LongBench)

### The Big Picture
This is an experiment in whether "linear-time sequence models" can match or beat Transformers while being dramatically faster and more efficient.

---

## Part 10: The Librarian's Note

This document was created as part of the ongoing SOP improvements (see [V4_HANDOFF.md](V4_HANDOFF.md) "SOP Self-Improvement" section).

**For future librarians:**
- If external researchers ask "what are you building?", point them here first
- If someone is confused about concepts vs implementation, say: "Start with ONBOARDING.md (concepts), then V4_DESIGN.md (how we built it)"
- Keep this document updated when Phase 3/4 change understanding

**Contributing to this doc:**
- Use clear mental models and analogies
- Link to source papers and repos
- Cite where Phase results support design choices
- Keep external references current (repos change, papers stay the same)

---

## Quick Navigation

**If you're new:**
1. Read Part 1 (Architectures)
2. Read Part 2 (Why Combine)
3. Skim Part 3 (Implementation results)
4. Then open [README.md](README.md) and [V4_DESIGN.md](V4_DESIGN.md)

**If you're implementing:**
1. Skim Part 1-2 for context
2. Read Part 3-4 for current state
3. Jump to [V4_STRATEGY.md](V4_STRATEGY.md) for your specific task
4. Reference [hybrid_v4_ratio.py](hybrid_v4_ratio.py) for code patterns

**If you're debugging:**
1. Read Part 6 (Glossary)
2. Check [V4_BUILD_LOG.md](V4_BUILD_LOG.md) for similar issues
3. Read [V4_HANDOFF.md](V4_HANDOFF.md) audit section

**If you're evaluating:**
1. Read Part 2 (Why Combine)
2. Read Part 3 (Phase 2 Results)
3. Check [CHANGELOG.md](CHANGELOG.md) for progression

---

**Last Updated:** 2026-01-10  
**For Questions:** See [V4_HANDOFF.md](V4_HANDOFF.md) "When Stuck" section
