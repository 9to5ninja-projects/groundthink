# V5 Gating: GPT-2 Cross-Comparison Benchmark

**Purpose:** Define the validation criteria for V5 release. V4 must demonstrate competitive performance against GPT-2 at matching parameter counts before proceeding.

**Created:** 2026-01-10  
**Status:** PLANNING (V4 must complete first)

---

## Context

We are at the stage of validating our 8M model and planning for 30M. We have a novel architecture (Mamba2+RWKV6) and we are building a stateful conversational model.

One of the key validation steps is to compare against a known baseline. GPT-2 is a well-known transformer model that has been extensively studied. Comparing our model against GPT-2 at the same parameter count can give us an idea of how our architecture performs relative to a transformer baseline.

**Key Difference:** Our architecture is stateful and designed for long contexts, while GPT-2 is a transformer with a fixed context window and no inherent state beyond that window. Benchmarks must be fair and highlight the strengths of both models.

---

## The GPT-2 Benchmark Strategy

### Parameter-Matched Comparisons

| Our Model | Comparable GPT-2 Variant | Purpose |
|-----------|--------------------------|---------|
| 3.5M | Train custom ~3M GPT-2 | Architecture sanity check |
| 8M | Train custom ~8M GPT-2 | Performance baseline |
| 30M | Train custom ~30M GPT-2 | Scaling efficiency test |
| 125M | Official GPT-2 Small (124M) | Industry benchmark |

### Why This Matters for GroundThink

1. **Establishes Realistic Expectations:** Is your 8M model learning at the same rate as a transformer with same compute?

2. **Architecture Efficiency Metric:** Your Mamba2+RWKV6 fusion claims efficiency advantages—prove it.

3. **Identifies Training Issues:** If GPT-2 outperforms yours significantly at same params/compute, you have optimization problems.

4. **Context Length Comparison:** Does your architecture actually outperform on long context at same scale?

---

## Benchmark Categories

### 1. Standard Language Modeling
- **Datasets:** WikiText-103, Penn Treebank
- **Metric:** Perplexity
- **Purpose:** Compare basic language modeling ability

### 2. Conversational Ability
- **Datasets:** DailyDialog, PersonaChat
- **Metrics:** Perplexity, BLEU, human evaluation
- **Purpose:** Test dialogue generation quality

### 3. Statefulness (GroundThink Advantage)
- **Tests:** Custom state persistence tests (S0-S4, C1-C6)
- **Note:** GPT-2 is not designed for statefulness beyond its context window
- **Purpose:** Demonstrate where our architecture excels

---

## Implementation Plan: Custom GPT-2 Training

### Architecture Design
- Use the **same tokenizer** as our model (BPE, GPT-2 tokenizer)
- Design GPT-2 with same context length (or as close as possible)
- Adjust layers, hidden size, and attention heads to match parameter count

### Training Protocol
- Train on the **same data** for the **same number of steps**
- Use same batch size and learning rate schedule (or known-good GPT-2 schedule)
- Fair comparison requires identical training conditions

### Evaluation Protocol
- Evaluate on the **same validation sets** (language modeling and conversational)
- Test GPT-2 on statefulness tests within its context window
- Document where each architecture excels

---

## Resource Considerations

### Recommended Approach
| Scale | Approach | Effort |
|-------|----------|--------|
| 3.5M | Train custom GPT-2 | Low |
| 8M | Train custom GPT-2 | Low |
| 30M | Train custom GPT-2 | Medium |
| 125M | Use official GPT-2 Small | None (pretrained) |

### Alternative (If Resources Limited)
- Compare 3.5M/8M models with 124M GPT-2 Small, adjusting for parameter count
- Use scaling laws to estimate GPT-2 performance at 3.5M/8M (less accurate)
- Compare perplexity per parameter

**Decision:** Training GPT-2 models at 3.5M, 8M, and 30M is worth the effort for accurate comparison.

---

## V5 Gate Criteria

**V5 PASS Requirements:**
1. ✅ All V4 models (3.5M, 8M, 30M, 125M) pass graduation tests
2. ✅ GPT-2 baselines trained at matching scales
3. ✅ GroundThink matches or exceeds GPT-2 on language modeling perplexity
4. ✅ GroundThink significantly outperforms GPT-2 on statefulness tests
5. ✅ Long-context performance advantage demonstrated

**V5 FAIL Triggers:**
- ❌ GPT-2 significantly outperforms GroundThink at same scale (optimization problem)
- ❌ No measurable statefulness advantage over GPT-2 (architecture hypothesis failed)

---

## Open Questions

1. What specific GPT-2 architecture dimensions for 3.5M, 8M, 30M?
2. Which statefulness tests are fairest for comparison?
3. Should we include inference speed comparisons?
4. What constitutes "significantly outperforms" (threshold)?

---

## References

- GPT-2 Paper: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- GPT-2 Small: 124M parameters, 12 layers, 768 hidden, 12 heads
- GPT-2 Medium: 355M parameters, 24 layers, 1024 hidden, 16 heads

---

## Implementation Details

### Step 1: Custom Mini-GPT-2 Configurations

```python
# Simplified GPT-2 configurations matching our scales
gpt2_configs = {
    "3.5M": {
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 256,
        "d_ff": 1024,
        "vocab_size": 50257,  # GPT-2 vocab
        "max_position": 1024  # Match our context length
    },
    "8M": {
        "n_layers": 6,
        "n_heads": 6, 
        "d_model": 384,
        "d_ff": 1536,
        "vocab_size": 50257,
        "max_position": 1024
    },
    "30M": {
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_ff": 3072,
        "vocab_size": 50257,
        "max_position": 1024
    }
}
```

### Training Protocol
- Same dataset as ours (or a standardized subset)
- Same tokenizer (critical for fair comparison)
- Same training steps/batch size/compute budget
- Same evaluation metrics

---

### Step 2: The 4-Way Comparison Matrix

For each model size, compare:

| Metric | Our Model | GPT-2 (custom) | Ideal Outcome |
|--------|-----------|----------------|---------------|
| Perplexity (validation) | | | Ours ≤ GPT-2 |
| Training Speed (tokens/sec) | | | Ours > GPT-2 (efficiency claim) |
| Memory Usage | | | Ours < GPT-2 (state-space advantage) |
| Context Length Scaling | | | Ours degrades slower |
| Stateful Task Accuracy | | | Ours >>> GPT-2 (our specialty) |

---

### Step 3: Specialized Benchmark Categories

Since GPT-2 won't have our state architecture, create two benchmark categories:

#### Category A: Transformer-Friendly Tasks (GPT-2 should do well)
- Next token prediction
- Short-range coherence
- Single-turn Q&A

#### Category B: State-Specialized Tasks (Where we should dominate)
- Multi-turn dialogue coherence
- Long-context information retrieval
- Conversation state tracking
- Context-dependent persona maintenance

---

## Critical Insights We'll Gain

1. **Architecture Efficiency:** Is Mamba2+RWKV6 more parameter-efficient than attention?
2. **Scaling Behavior:** Do we follow similar scaling laws or diverge?
3. **Statefulness Value:** How much does our state mechanism actually help?
4. **Training Dynamics:** Are there optimization differences at same scale?

---

## Key Metrics

**Note:** Detailed metric implementations are in [V4_TESTING.md](V4_TESTING.md#required-analysis-metrics) (sections 5-7).

### 1. Architecture Efficiency Ratio

Calculate: `(Our Perplexity) / (GPT-2 Perplexity)`

| Ratio | Interpretation |
|-------|----------------|
| < 1.0 | We're more parameter-efficient |
| ≈ 1.0 | Parity with transformers |
| > 1.0 | We're less efficient (need architecture fixes) |

### 2. Context Window Scaling Curve

Plot perplexity vs context length for both architectures:

```
GPT-2: PPL increases sharply after 512 tokens
Ours:  PPL increases gradually to 2048+ tokens (if architecture works)
```

### 3. Training Dynamics Comparison

- Does our model converge faster/slower?
- Are loss curves smoother/more chaotic?
- Does it overfit less/more?

---

## Practical Implementation

### Training Commands

```bash
# Our training command
python train_v4.py --model GF-MH --params 8M --context 1024

# GPT-2 training command (using minimal GPT-2 code)
python train_mini_gpt2.py --config gpt2_8M.json --context 1024

# Evaluation command
python benchmark.py --model groundthink_8M --baseline gpt2_8M --tasks stateful_tasks.json
```

### Recommended GPT-2 Implementations

| Scale | Implementation | Notes |
|-------|----------------|-------|
| 3.5M, 8M, 30M | NanoGPT (Karpathy) | Perfect for small-scale training |
| 125M | HuggingFace GPT-2 | Use pretrained weights |
| Custom | Manual implementation | For exact parameter matching |

### Validation Dataset (Standardized 100MB)

Use the SAME corpus for ALL comparisons:

| Content Type | Percentage | Purpose |
|--------------|------------|---------|
| Conversational (multi-turn) | 50% | State tracking |
| Long documents | 30% | Context testing |
| Code | 20% | Structured thinking |

---

## Execution Milestones

| Milestone | Action |
|-----------|--------|
| **8M trained** | Run GPT-2-8M comparison (Tasks 62-65) |
| **Before 30M** | If 8M doesn't match GPT-2-8M within 30%, fix architecture first |
| **At 125M** | Compare with official GPT-2 Small |

**See [V4_STRATEGY.md](V4_STRATEGY.md) Tasks 62-65 for implementation.**

---

## Decision Matrix: Expected Outcomes

### Scenario 1: We Beat GPT-2

| Outcome | Action |
|---------|--------|
| Both efficiency AND stateful tasks | ✅ Scale immediately |
| Stateful tasks only, equal efficiency | ✅ Validates architecture choice |
| Efficiency only, equal on tasks | ⚠️ Improve state mechanisms |

### Scenario 2: We Underperform GPT-2

| Outcome | Action |
|---------|--------|
| Lower efficiency AND worse tasks | ❌ Architecture needs rework |
| Equal efficiency, worse on state tasks | ❌ State implementation is broken |
| Worse efficiency, better state tasks | ⚠️ Acceptable trade-off if statefulness is critical |

---

## The "Blind Taste Test"

Create a Turing-style test with outputs from:
1. Our 8M model
2. GPT-2 8M (custom trained)
3. GPT-2 124M (as upper bound)

Have evaluators rank:
- Coherence in multi-turn conversations
- Fact consistency across long contexts
- Naturalness of responses

This gives qualitative validation beyond metrics.

---

## Why This Is Critical for GroundThink

Our architecture (Mamba2+RWKV6) is **novel and unproven at scale**. The transformer has 7 years of optimization and known scaling laws. We need to:

1. **Prove efficiency claims** (linear scaling, state efficiency)
2. **Validate architectural choices** before investing in 30M/125M training
3. **Establish credibility** for our approach

### The Reality Check

> If we can't beat or match GPT-2 at 8M parameters with 7 years less research, we need to seriously reconsider our architecture before scaling. This isn't about perfection—it's about proving our fundamental approach has merit.

---

## Alternative: OPT-125M Comparison

If training custom GPT-2s is too heavy:
- Use **OPT-125M** as the 125M benchmark (open weights, similar architecture)
- But for 8M/30M, we really should train custom models

---

## The Real Competition Landscape

### Forget GPT-3 Comparisons (For Now)

GPT-3 175B cost millions to train. It's irrelevant for our scale. What matters are architectures we can actually train with our resources.

> A model is just a mathematical blueprint—the "weird science" is the alchemy of data, compute, and optimization.

### The Actual Proprietary Benchmarks That Matter

| Model | Our Comparison Point | Why It's Fair | Reality Check |
|-------|---------------------|---------------|---------------|
| GPT-2 (124M) | Our 125M | Same params, open weights, well-studied | Must-match baseline |
| Pythia (70M-410M) | Our 125M | Open weights, same data (The Pile), documented scaling | Better than GPT-2 for comparison |
| TinyLlama (1.1B) | Our 125M | Aggressively optimized, modern training | Shows how far small models can go |
| Mistral 7B | Not directly | Architectural inspiration | Study their choices, don't compare performance |

### The Math vs Money Reality

A 175B parameter model trained on 1T tokens will beat our 125M model. That's math:
- More parameters = more capacity
- More tokens = better generalization
- More compute = better optimization

**Our question becomes:** "Does our architecture give us an advantage at the same parameter/compute budget?"

---

## Tiered Benchmarking Strategy

### Tier 1: Must Beat (Architecture Validation)

| Target | Purpose |
|--------|---------|
| GPT-2 Small (124M) on same compute budget | Prove basic efficiency |
| Custom-trained Pythia-125M on our data | Fair comparison |
| Our own previous version | Regression testing |

### Tier 2: Aspire to Match (Efficiency Goal)

| Target | Purpose |
|--------|---------|
| TinyLlama-1.1B on specific efficiency metrics | Tokens/second, memory |
| Mamba (original) on state retention tasks | State mechanism validation |
| RWKV-5/6 on long-context coherence | Architecture family comparison |

### Tier 3: Study, Don't Compare (Reality Check)

| Target | Purpose |
|--------|---------|
| GPT-3.5/4, Claude, Gemini | Commercial products, 100-1000x our compute |
| Mistral 7B, Llama 2/3 7B | Study architectural choices |

---

## The "Math Advantage" We Need to Prove

Our architecture claims:

| Claim | Traditional Transformer | Our Architecture |
|-------|------------------------|------------------|
| Context scaling | O(n²) quadratic | O(n) linear |
| State retention | Attention window only | Persistent state |
| Training efficiency | O(n²) per step | O(n) per step |

**Prove these mathematically first before claiming performance advantages.**

### Theoretical Advantage Calculation

```python
# Theoretical advantage calculation
your_flops_per_token = n * d_model  # Mamba/RWKV claim
transformer_flops_per_token = n² * d_model  # Standard attention

# At 2048 context length:
transformer_cost = 2048² = 4.2M units
your_cost = 2048 = 2K units

# That's 2100x theoretical advantage
# But reality: Do you achieve same quality with 2100x less compute?
```

---

## The Proprietary Edge: What We Actually Control

| What We Can Control | What Others Control |
|---------------------|---------------------|
| Architecture efficiency | More compute |
| Data quality/curation | More data |
| Training optimizations | More engineers |
| Specialized capabilities | General capabilities |

**Our advantage must be:** Doing something at 125M that transformers need 1B+ to do, or doing it much cheaper.

---

## Concrete Action Plan: Validate Ourselves First

### Phase 1: Internal Validation (Next 2 Weeks)

```python
# Don't compare to others yet. Compare to YOURSELF:
1. 8M vs 3.5M: 2x params → How much better?
2. Project to 125M: Does scaling curve predict useful performance?
3. If 125M prediction looks competitive with GPT-2-124M → Continue
4. If not → Fix architecture at 8M scale
```

### Phase 2: Open-Source Comparison (After 125M trained)

```bash
# Only after you have 125M trained:
python compare.py \
  --your_model groundthink-125M \
  --baselines gpt2-124M pythia-125M \
  --metrics "state_retention,context_length_scaling,efficiency" \
  --ignore "general_benchmarks"  # You'll lose on MMLU, etc.
```

### Phase 3: Find Your Niche (Strategic Advantage)

What can our 125M do that others' can't at same size?

| Potential Niche | Architecture Advantage |
|-----------------|------------------------|
| 100K context with minimal memory | Mamba/RWKV linear scaling |
| Real-time conversation with persistent state | Native state mechanism |
| Edge deployment on limited hardware | Lower compute per token |

---

## The "Weird Science" We Should Actually Measure

Beyond standard benchmarks, measure what makes our architecture unique:

1. **State persistence across sessions** — Can we resume conversations?
2. **Memory efficiency at long context** — VRAM usage at 8K, 16K, 32K tokens
3. **Inference latency** — Time to first token, time per token
4. **Training efficiency** — Loss per FLOP, not just loss per step

---

## Custom Metrics for Our Architecture

### 1. Compute-Efficiency Ratio (CER)

```
Our CER = (Our Performance) / (Our Compute)
GPT-2 CER = (GPT-2 Performance) / (GPT-2 Compute)

If Our CER > GPT-2 CER: We have architectural advantage
If Our CER < GPT-2 CER: Fix or abandon
```

### 2. "Useful Context Window" Metric

| Architecture | Useful Context |
|--------------|----------------|
| Standard transformers | 0.25 × trained length |
| Our claim | 1.0 × trained length (or more) |

**Test:** Train on 2K context, test on 8K, 16K, 32K.
Does performance degrade gracefully or catastrophically?

### 3. State Persistence Score

Create a metric: How many conversation turns before key fact is lost?

| Model Type | Expected Turns |
|------------|----------------|
| Transformer baseline | 3-5 turns typically |
| Our goal | 50+ turns |

---

## The Brutal Proprietary Truth

### What Investors/Management Want to Hear
- "We beat GPT-2 at same parameter count"
- "Our model is 10x more efficient on long context"
- "We can do real-time conversation on a phone"

### What We Need to Know Internally
- "Our architecture works as theoretically predicted"
- "We understand the scaling laws for our approach"
- "We can predict what 500M/1B will do"

---

## The Only Comparisons That Matter Right Now

### For Our 8M Model
| Comparison | Purpose |
|------------|---------|
| Previous checkpoint | Regression testing |
| Ablation studies (Mamba-only, RWKV-only, fusion) | Component contribution |
| Theoretical minimum | What should be possible |

### For Our 30M Model (when we get there)
| Comparison | Purpose |
|------------|---------|
| GPT-2 Small (124M) adjusted for parameter efficiency | Scaled baseline |
| Pythia-70M | Closer match |
| Our own 8M model | Scaling validation |

### For Our 125M Model
| Comparison | Purpose |
|------------|---------|
| GPT-2 Small (124M) | Must match or beat |
| TinyLlama-1.1B on efficiency metrics only | Efficiency ceiling |
| Commercial APIs | Cost comparison, not quality |

---

## The "Blueprint" Philosophy

A model architecture is indeed a blueprint. The building (trained model) requires:

| Component | Meaning |
|-----------|---------|
| Materials | Data quality |
| Labor | Compute time |
| Craftsmanship | Training skill |
| Design efficiency | Architecture |

**Our edge must be in #4 because we can't compete on #1-3 with big players.**

---

## The Red Line

If after 3 iterations at 8M we're not:

- ❌ Matching GPT-2-8M efficiency within 30%
- ❌ Showing clear state retention advantage
- ❌ Having predictable scaling curves

**Then reconsider the architecture fundamentally. No amount of scaling will fix a broken blueprint.**

---

## Final Proprietary Advice

> Stop thinking about beating others. Think about:
> 1. Does our math work as predicted?
> 2. Can we build something useful at our scale?
> 3. What can we do that others can't at same cost?

**Our value proposition isn't "better than GPT-3."** It's:
- "Good enough at X with 100x less cost"
- OR "Capable of Y that others can't do at any cost"

### Find Our X or Y

| Potential X or Y | Why It Matters |
|------------------|----------------|
| Long context (100K+) | Linear scaling advantage |
| Real-time inference | Lower latency per token |
| Edge deployment | Smaller memory footprint |
| State persistence | Native architecture feature |

**Define our X or Y, then build validation specifically for it.**

---

## Open Question

> What's our X or Y? Long context? Real-time? Edge deployment? State persistence?

*This is the strategic decision that defines our positioning.*

---

*This document will be expanded as V4 completes and V5 planning begins.*
