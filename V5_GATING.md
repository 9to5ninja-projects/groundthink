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

## Baseline Data Sources

### Official References
| Source | What It Provides | Link |
|--------|------------------|------|
| Pythia | Training logs for 70M-12B models on The Pile | https://github.com/EleutherAI/pythia |
| Papers with Code | GPT-2 validation perplexity curves | GPT-2 page |
| HuggingFace | GPT-2-small reported perplexity | Model cards |
| NanoGPT | Clean GPT-2 reimplementation | https://github.com/karpathy/nanoGPT |
| lm-evaluation-harness | Standard benchmark suite | https://github.com/EleutherAI/lm-evaluation-harness |

### Reference Numbers (Pythia)
| Model Size | Final Loss | Training Hours | Data Tokens |
|------------|------------|----------------|-------------|
| 70M | 2.85 | 8h on 8×A100 | 300B |
| 160M | 2.70 | 18h on 8×A100 | 300B |
| 410M | 2.45 | 48h on 8×A100 | 300B |

**GPT-2 Small Reproduction:** ~2.2 loss on OpenWebText, ~1 week on 8×V100

---

## The Matching Recipe

### Config Alignment
```python
matching_config = {
    "token_count": 2_000_000_000,  # 2B tokens total
    "batch_size": 512,
    "context_length": 1024,
    "vocab_size": 50257,  # GPT-2's vocab
    "learning_rate_schedule": "cosine",
    "warmup_steps": 2000,
    "max_lr": 3e-4,
    "weight_decay": 0.1,
    "optimizer": "AdamW",
    "betas": (0.9, 0.95),
    "grad_clip": 1.0
}
```

### Critical: Same Data Order
```python
# For true comparability, use same random seed and batch order
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Save batch indices
batch_indices = np.random.permutation(len(dataset))
np.save('batch_order.npy', batch_indices)

# Both models load same batch order
gpt2_loader = DataLoader(dataset, sampler=SubsetRandomSampler(batch_indices))
your_loader = DataLoader(dataset, sampler=SubsetRandomSampler(batch_indices))
```

### GPT-2 Architecture Configs (All Scales)

**Parameter Calculation:**
```python
def gpt2_params(vocab=50257, max_pos=1024, n_layers=6, d_model=384):
    embeddings = vocab * d_model
    positional = max_pos * d_model
    attn = 4 * d_model * d_model  # Q,K,V + output projection
    mlp = 2 * d_model * (4 * d_model)  # FFN up/down
    layer_norm = 2 * d_model
    per_block = attn + mlp + layer_norm
    return embeddings + positional + (n_layers * per_block)
```

**Configs:**
```python
gpt2_configs = {
    "3.5M": {
        "n_layers": 3,
        "n_heads": 4,
        "d_model": 256,
        "d_ff": 1024,
        "total_params": 3_512_000,
        "notes": "Smallest viable GPT-2 with full vocab"
    },
    "8M": {
        "n_layers": 6,
        "n_heads": 6,
        "d_model": 384,
        "d_ff": 1536,
        "total_params": 8_112_000,
        "notes": "Balanced small model"
    },
    "30M": {
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 512,
        "d_ff": 2048,
        "total_params": 29_875_000,
        "notes": "Matches GPT-2 scaling"
    },
    "125M": {
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_ff": 3072,
        "total_params": 124_000_000,
        "notes": "Official GPT-2 Small (direct benchmark)"
    }
}
```

---

## Three Pillars of Fair Comparison

### Pillar 1: Compute-Equivalent
- Same GPU hours (A100-hours)
- Same FLOPs count: `params × tokens × 6`
- Same memory budget

### Pillar 2: Data-Equivalent
- Same token count (2B tokens)
- Same data distribution (webtext:books:code ratio)
- Same preprocessing pipeline

### Pillar 3: Optimization-Equivalent
- Same hyperparameter tuning budget
- Same early stopping criteria
- Same validation set

---

## Evaluation Suite

```python
evaluation_suite = {
    "standard_metrics": {
        "wikitext_ppl": "Wikitext-103 perplexity",
        "lambada": "LAMBADA accuracy",
        "hellaswag": "HellaSwag accuracy"
    },
    "specialized_metrics": {
        "state_retention": "Fact recall at 5/10/20/50 turns",
        "long_context": "2K→32K degradation curve",
        "conversation_coherence": "Multi-turn dialogue tests"
    },
    "efficiency_metrics": {
        "tokens_per_second": "Inference speed",
        "memory_footprint": "VRAM usage",
        "context_scaling": "2K/4K/8K context performance"
    }
}
```

---

## Statefulness Tests (Fair to Both Architectures)

### Tier 1: Short-Term Memory
```python
short_term_tests = {
    "3_turn_memory": {
        "template": "My favorite color is {color}. " * 3 + "What's my favorite color?",
        "metric": "Accuracy of color recall after 3 repetitions"
    },
    "contextual_correction": {
        "template": "I like apples. Actually, I prefer oranges. What do I like?",
        "metric": "Accuracy of updated preference"
    }
}
```

### Tier 2: Medium-Term Memory
```python
medium_term_tests = {
    "interrupted_dialogue": {
        "template": "Set: I'm going to Paris.\n[10 unrelated sentences]\nWhere am I going?",
        "metric": "Recall after distraction"
    },
    "multi_fact_composition": {
        "template": "Alice is 30. Bob is 25. Who is older?",
        "metric": "Accuracy of composed facts"
    }
}
```

### Tier 3: Conversational
```python
conversational_tests = {
    "persona_consistency": {
        "template": "You're a helpful chef. [5 cooking Q&A turns] Now help me with coding?",
        "metric": "Does it maintain chef persona inappropriately? (shouldn't)"
    },
    "topic_threading": {
        "template": "Let's discuss dogs. [talk about breeds] Now compare to cats.",
        "metric": "Naturalness of topic transition"
    }
}
```

---

## Inference Speed Protocol

### What to Measure
```python
speed_metrics = {
    "prefill_time": {
        "method": "Time to process 1024 tokens (first forward pass)",
        "units": "seconds",
        "repeats": 100
    },
    "generation_speed": {
        "method": "Time per token during 256-token generation",
        "units": "tokens/second",
        "repeats": 10
    },
    "memory_usage": {
        "method": "Peak GPU memory at 1024 context",
        "units": "MB",
        "notes": "torch.cuda.max_memory_allocated()"
    },
    "context_scaling": {
        "method": "Time vs context length [256, 512, 1024, 2048]",
        "plot": "Should show linear vs quadratic growth"
    }
}
```

### Fair Comparison Rules
- Same hardware (A100, same cooling conditions)
- Same batch size (batch=1 for inference, batch=8 for training)
- Same precision (float16 or bfloat16)
- Same software stack (PyTorch version, CUDA version)

### Compute-Efficiency Ratio (CER)
```
CER = (Performance Metric) / (Inference Time)

Example:
  GPT-2: Perplexity = 2.8, Time = 100ms → CER = 28
  Ours:  Perplexity = 3.0, Time = 40ms  → CER = 75

Higher CER = Better efficiency
```

---

## Decision Thresholds

### Statistical Significance
```python
# For any metric, run 10 trials
# Significant if: |mean1 - mean2| > 2 * sqrt(σ1² + σ2²)

significant_improvement = {
    "loss_perplexity": ">10% relative improvement",
    "accuracy": ">5% absolute improvement",
    "speed": ">20% faster at same quality",
    "memory": ">30% reduction"
}
```

### Practical Thresholds (8M vs GPT-2-8M)
| Metric | "Good Enough" | "Promising" | "Excellent" |
|--------|---------------|-------------|-------------|
| Validation Loss | Within 15% | Within 5% | Better |
| Statefulness Score | 10% better | 25% better | 50% better |
| Inference Speed | 20% faster | 50% faster | 2x faster |
| Memory Usage | Equal | 20% less | 50% less |

### Graduation Criteria (8M → 30M)
**Must meet AT LEAST one "Excellent" OR two "Promising":**
```
Example passes:
  1. Statefulness: 30% better + Speed: 40% faster → GO
  2. Loss: 8% better + Memory: 40% less → GO

Example fails:
  1. All metrics "Good Enough" → FIX FIRST
  2. Loss: 20% worse (even with speed) → FIX FIRST
```

### Red Line Thresholds (STOP and fix)
```python
stop_conditions = {
    "loss_ratio": "Your loss > 1.3 × GPT-2 loss",
    "statefulness": "Your score < GPT-2 score",
    "training_instability": "Loss spikes > 2x average",
    "scaling_violation": "8M not better than 3.5M by expected margin"
}
```

### Decision Logic
```python
def should_scale(metrics):
    # Rule 1: Not significantly worse on loss
    if metrics["loss_ratio"] > 1.15:
        return False, "Loss too high"
    
    # Rule 2: Show advantage somewhere
    advantages = sum(v > 1.1 for v in [
        metrics["statefulness_ratio"],
        metrics["speed_ratio"],
        metrics["memory_ratio"]
    ])
    
    if advantages >= 2:
        return True, f"Has {advantages} clear advantages"
    elif advantages == 1 and metrics["loss_ratio"] < 1.05:
        return True, "Balanced trade-off"
    else:
        return False, "No clear advantage"
```

---

## The Bottom Line

**For architecture to be viable (8M):**
- Not >15% worse on loss than GPT-2-8M
- Show >20% advantage on at least one of: statefulness, speed, memory, context scaling

**For architecture to be exciting:**
- Match or beat GPT-2-8M on loss
- Show >40% advantage on statefulness
- Show >50% advantage on efficiency metrics

**Key insight:** If we're within 10% of GPT-2 on standard metrics but 2x better on statefulness, that's a breakthrough for conversational AI.

---

## Reverse Testing: GPT-2 in Our Framework

**Purpose:** Validate our training loop by training GPT-2 in it.

```python
# Step 1: Implement GPT-2 in your codebase
class GPT2InYourFramework(nn.Module):
    # Recreate GPT-2 exactly in your training loop

# Step 2: Train with YOUR optimizers, data pipeline
python train_v4.py --model=gpt2_8M --framework=yours

# Step 3: Compare results with reference GPT-2
```

**Diagnostic Matrix:**
| Result | Interpretation |
|--------|----------------|
| Your GPT-2 ≈ Reference (±5%) | Framework is correct |
| Your GPT-2 worse (>10%) | Training loop has bugs |
| Your GPT-2 better | Document the improvements |

---

## Smoke Test Protocol

Run before full training:

```python
# 1. 1% data test (20M tokens)
train_1_percent(your_8M, gpt2_8M)
# Expected: Both should show similar loss decrease slope

# 2. Overfitting test
train_on_tiny_set(your_8M, gpt2_8M, size=1000)
# Expected: Both should reach near-zero loss

# 3. Gradient norm test
check_gradient_norms(your_8M, gpt2_8M)
# Expected: Similar gradient magnitudes
```

---

## Scaling Laws Validation

```python
# Expected scaling law: L = a * N^(-b) + c
# Where N = parameters

# If your scaling is better (steeper b):
#   → You're more parameter-efficient

# If your scaling is worse:
#   → Architecture has fundamental limitations
```

**Plot loss vs params for both models on log-log scale. Curves should be parallel.**

---

## Pragmatic Approach

### Step 1: Train GPT-2 Baseline
1. Train GPT-2-8M in NanoGPT on 1B tokens of your data
2. Record final loss/perplexity
3. That's your baseline to beat

### Step 2: Train Our Model
1. Train your 8M with identical settings
2. Compare numbers
3. If within 10%, proceed to 30M
4. If not, debug at 8M

### Interpretation Guide
| Comparison | Interpretation |
|------------|----------------|
| Our 8M loss 2.8 vs GPT-2 2.7 | ✅ Good enough |
| Our 8M loss 3.5 vs GPT-2 2.7 | ❌ Architecture problems |
| Our 8M loss 2.5 vs GPT-2 2.7 | 🎉 Something special |

**Threshold:** Within 20% at 8M → proceed. >20% difference → stop and debug.

---

## Validation Checklist Per Scale

### 3.5M Models (Both)
- [ ] Can overfit 10,000 tokens (memorization test)
- [ ] Validation loss plateaus after 500M tokens
- [ ] Training curves are smooth (no spikes)

### 8M Models
- [ ] Outperforms 3.5M by expected margin (scaling law)
- [ ] Shows emergent simple reasoning
- [ ] Efficiency metrics are stable
- [ ] Within 20% of GPT-2-8M loss

---

## Open Questions

1. ~~What specific GPT-2 architecture dimensions for 3.5M, 8M, 30M?~~ → Answered: GPT-2 Architecture Configs section
2. ~~Which statefulness tests are fairest for comparison?~~ → Answered: Statefulness Tests section (Tier 1-3)
3. ~~Should we include inference speed comparisons?~~ → Yes: Inference Speed Protocol section
4. ~~What constitutes "significantly outperforms" (threshold)?~~ → Answered: Decision Thresholds section
5. **What's our X or Y?** Long context? Real-time? Edge deployment? State persistence?

**Strategic question #5 defines our positioning.**

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

*This document defines V5 gate criteria. Build diagnostic tools (Tasks 52-61), then validate (Tasks 62-66).*
