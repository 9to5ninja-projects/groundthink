# Scaling Milestones: Strategic Framework for Parameter Growth

**Purpose:** Define what each scale threshold should achieve, what to measure, and confidence criteria for graduation between scales.

**Version:** 4.10-Alpha  
**Created:** 2026-01-10  
**Type:** Strategic Foundation Document  
**Related Docs:** V4_STRATEGY.md (phases), CANARY_TESTS.md (testing), V4.5_VALIDATION.md (validation entries)

---

## Philosophy: Each Scale is an Experimental Regime

Each parameter count isn't just a bigger model; it's a different experimental regime with distinct learning objectives.

**Core Principle:** Scaling isn't automatic—each jump should unlock new capabilities or reveal scaling laws that inform the next step. The goal is to build a **validation ladder** where each rung provides proof of concept and guides the next step.

**Questions Each Scale Answers:**
- 3.5M: "Does the training system work at all?"
- 8M: "Can the architecture learn real patterns?"
- 30M: "Do scaling laws predict? Do capabilities emerge?"
- 125M: "Is this production-ready? What comes next?"

---

## 3.5M Parameters: Sanity Check & Architecture Debug

### Purpose
Validate data pipeline, training loop, and architecture before scaling. This is pure de-risking.

### What to Achieve

**State Space Fundamentals (NEW — PRIORITY)**
Before any capability testing, verify the state machinery works:
- S0: State vectors exist with correct shapes (RWKV + Mamba)
- S1: State initialization is healthy (non-zero, reasonable magnitude)
- S2: State evolves with different inputs (not frozen)
- S3: State is deterministic (same input → same state)
- S4: Both components contribute (activation variance ratio <100x)

**Why State Space First:**
- At 3.5M, you cannot test capabilities (too small for complex reasoning)
- But you CAN verify the state machinery is functional
- If state spaces are broken here, they'll be broken at 30M
- This is the cheapest place to catch architecture bugs

**See [CANARY_TESTS.md](CANARY_TESTS.md#s0-s4-state-space-fundamentals-35m-only--required-first) for S0-S4 test implementations.**

**Proof of Life**
- Can the model overfit a tiny, memorizable batch (10-100 samples)?
- If not: training loop, loss function, or data loading is broken
- This is the first critical gate

**Data Pipeline Validation**
- Preprocessing, tokenization, batching flow without errors
- No data leaks between train/val
- Correct shapes throughout forward pass
- **BPE tokenization working** (not char-level)

**Baseline Performance**
- Establish lower bound on validation set
- Compare to "random" baseline (naive token prediction)
- Anything better is progress

**Architecture Soundness**
- Gradients flow through all components
- Parameters update during training
- No shape mismatches or NaN errors
- **Component balance documented** (even if imbalanced at this scale)

### Confidence Criteria to Graduate (→ 8M)

✅ **State space tests pass (S0-S4)** — state machinery verified

✅ Training loss drops smoothly on a single batch (proves learning works)

✅ Model can overfit a small subset (10-100 samples, 100% accuracy possible)

✅ Validation loss < naive baseline (e.g., "always predict most common token")

✅ No major bugs in code or data pipeline

✅ Can checkpoint and resume training

✅ **BPE tokenization verified** (not char-level)

✅ **Component balance documented** (ratio and activation variance recorded)

**Tone:** This is a code/process confidence milestone, not a performance one. You're testing infrastructure AND state space health, not model quality.

---

## 8M Parameters: Proof of Concept & Pattern Discovery

### Purpose
Demonstrate core task learning on full dataset and begin structured experimentation with hyperparameters and variants.

### What to Achieve

**Meaningful Learning**
- Clear, non-trivial improvement on full training set
- For language models: next-token prediction loss substantially below random
- For classification: accuracy > baseline + 10+ percentage points
- Loss curves are smooth, predictable, reproducible

**Structured A/B Testing**
- Test hyperparameter variants: learning rates, warmup schedules, optimizer choices
- Test architectural variants: different fusion mechanisms, component ratios, gate designs
- At this scale, experiments are cheap—iteration is fast
- Document results in [V4_FUSION_MODELS.md](V4_FUSION_MODELS.md) style tables

**First-Order Data Patterns**
- Qualitative analysis: What is the model learning?
  - For language: basic syntax? Domain keywords? Sentence boundaries?
  - For specialized tasks: task-specific patterns?
- Human evaluation of sample outputs
- Error analysis: What does it get wrong consistently?

**Compute Budget Estimation**
- Measure steps-to-convergence
- Estimate wall-clock time to convergence
- Identify bottlenecks (data loading? CUDA? gradient computation?)
- Forecast compute needs for 30M based on scaling observed

### Confidence Criteria to Graduate (→ 30M)

✅ Validation loss/accuracy clearly improves over 3.5M model (≥5% improvement, or loss drop ≥0.5 nats)

✅ Improvement is reproducible across 2+ independent training runs

✅ Identified best-performing hyperparameter set (LR, warmup, optimizer, regularization)

✅ Human evaluation confirms learning relevant, domain-specific patterns (not random)

✅ **Have a hypothesis for 30M:** "At 8M, it's learning X. At 30M, we expect to see emergence of Y."

✅ Confidence in architectural choice: fusion mechanism works as intended, components are both active

**Tone:** This is the "does it work on real data?" milestone. You're proving the core concept before scaling up.

---

## 30M Parameters: Scaling Laws & Capability Emergence

### Purpose
Demonstrate that scaling laws are predictable and that valuable capabilities emerge at scale. This is where the research story becomes compelling.

### What to Achieve

**Quantify Scaling Laws**
- Plot loss vs. compute for 8M and 30M
- Calculate compute efficiency: loss improvement per TFLOP
- Do curves follow power law? (loss ∝ compute^{-α})
- If yes: can confidently extrapolate to 125M
- If no: investigate why (data saturation? architecture bottleneck?)

**Capability Emergence**
- Define 5-10 canary tasks specific to your domain:
  - Language model: multi-turn coherence, fact consistency, instruction following
  - Classification: nuanced distinction between similar categories, rare-class performance
  - Specialized: domain-specific reasoning, technical accuracy
- Measure quantitatively where possible (accuracy, F1, custom metrics)
- Document "8M achieved X%, 30M achieved Y%" for each
- Only graduate if ≥3 canaries show meaningful improvement (≥10-20% relative improvement)

**Robust Train/Validation Analysis**
- Track train vs. val divergence carefully
- The gap will widen; this is normal
- Establish regularization is working:
  - Dropout effective?
  - Weight decay effective?
  - Early stopping rule is working?
- Analyze failure modes: where does 30M fail more than 8M? (Indicates data gaps)

**Blind Testing & Human Evaluation**
- Formal blind A/B tests with human evaluators
- "Which output is better: 8M or 30M?" on 50+ samples
- Score on: relevance, coherence, accuracy, style-appropriateness
- Target: statistically significant preference for 30M (p < 0.05)

### Confidence Criteria to Graduate (→ 125M)

✅ Scaling curves are predictable and fit power law well (R² > 0.95 on log-log plot)

✅ Model achieves ≥3 target "canary capabilities" at >80% of human-level performance

✅ Blind A/B tests show statistically significant preference for 30M over 8M

✅ Scaling law predicts 30M performance accurately based on 8M (within ±5%)

✅ High confidence in training recipe: know exactly what hyperparameters work and why

✅ No concerning instabilities: training stable, loss curves smooth, no unexplained spikes

**Tone:** This is the "do we have something real?" milestone. You're validating the entire experimental framework before going to production scale.

---

## 125M Parameters: MVP & Scaling Finalization

### Purpose
Deliver a model that is demonstrably useful for the core task and finalize scaling predictions for future work.

### What to Achieve

**MVP (Minimally Viable Product) Performance**
- Model is good enough for:
  - Public demo (users can interact and see value)
  - Internal tool (solves a real problem, not just a toy)
  - Rigorous academic paper (publishable results)
- Performance clearly "useful" for the core task:
  - Language model: generates coherent, relevant multi-paragraph text
  - Classification: performs comparably to baselines on standard benchmarks
  - Specialized: solves the problem it was designed for with acceptable accuracy

**Finalize Scaling Laws**
- With 4 data points (3.5M, 8M, 30M, 125M), fit a reliable scaling law
- This is your key planning tool for future work:
  - "To reach 85% accuracy, we need X parameters"
  - "Each 2x parameter increase costs Y compute and improves performance Z%"
- Validate predictions: does the 125M model match the scaling law prediction?

**Comprehensive Evaluation**
- In-distribution validation: full validation set, multiple metrics
- Out-of-distribution tests: domain shift, adversarial examples, rare cases
- Long-context evaluation (if applicable): how does it perform on longer sequences?
- Extensive human evaluation:
  - 200+ prompts across diverse categories
  - Multiple evaluators (≥3) for inter-rater reliability
  - Structured rubric: coherence, accuracy, relevance, style, safety

**Failure Analysis & Data Gap Discovery**
- Systematic failure analysis: what does it get wrong?
  - Categorize failures: factual errors, logical errors, style errors, knowledge gaps?
  - Quantify: "X% of failures due to missing knowledge in training data"
- Identify highest-impact improvements:
  - Is performance limited by model size or data quality?
  - Would more data help? (Yes → next phase is data curation)
  - Would architectural changes help? (Yes → investigate in parallel)

### Confidence Criteria to Graduate (→ Production or Large-Scale Training)

✅ Meets pre-defined performance thresholds on key metrics (examples):
   - Language model: perplexity < 50 on benchmark, human eval score > 4/5
   - Classification: accuracy > 92%, F1 > 0.88 on all classes
   - Specialized: solves the core task with >85% correctness

✅ Scaling law is accurate: 125M prediction based on 3.5M/8M/30M falls within ±5% of actual performance

✅ Scaling law is reliable enough to predict 500M/1B: can confidently state "next checkpoint needs X more data or Y more compute"

✅ Human evaluation is statistically rigorous: ≥3 evaluators, inter-rater agreement κ > 0.7, sample size > 100

✅ Have a clear, prioritized list of limitations:
   - "Top 3 improvements: (1) Add finance domain data, (2) Improve instruction-following, (3) Reduce hallucination"
   - Not vague ("make it better") but specific ("add 50K finance documents with X characteristics")

✅ Confidence in data/compute roadmap: "To reach production quality at 500M, we need 2 more data collection cycles + 8 weeks on A100 cluster"

**Tone:** This is the "do we have something real to show?" milestone. You're proving the model is useful AND that you understand the scaling laws that got you here.

---

## Summary: Confidence Checklist

Before graduating from one scale to the next, ask:

### Technical Readiness
- [ ] Is training stable and reproducible?
- [ ] Do metrics improve predictably?
- [ ] Are all major bugs fixed?
- [ ] Can I resume training from checkpoints?

### Scientific Validity
- [ ] Have we learned something new about the problem/data/architecture?
- [ ] Do scaling predictions hold? (Can we predict next scale's performance within ±10%?)
- [ ] Are results reproducible across multiple runs?

### Capability Advancement
- [ ] Has the model unlocked a new, qualitatively different capability?
- [ ] Is it demonstrably better than the previous scale in a meaningful way?
- [ ] Does human evaluation support the quantitative improvements?

### Resource Forecasting
- [ ] Can we confidently predict compute cost for the next scale?
- [ ] Do we know the bottleneck (data? compute? architecture)?
- [ ] Can we estimate data needs for the next phase?

**If all four boxes are checked, graduate. Otherwise, extend the current scale or revisit the architecture.**

---

## Risk: What NOT to Do

🚫 **Don't jump scales without validation:** Skipping the 8M phase to go straight to 30M wastes compute and leaves you blind to fundamental problems.

🚫 **Don't rely on single runs:** Reproduce your best result at least once. Luck happens.

🚫 **Don't confuse scale with capability:** A bigger model that doesn't solve new problems isn't progress.

🚫 **Don't skip human evaluation:** Metrics can be misleading. Humans catch nuances metrics miss.

🚫 **Don't forget failure analysis:** The 125M model's failures are data for the next phase. Document them.

---

## How This Applies to GroundThink V4

**Current Status:** Planning 8M validation phase (Phase 3.9)

**Our Scaling Journey:**
- ✅ **3.5M (Phase 0-1):** Architecture debug, CUDA kernel verification, basic training
- 🟡 **8M (Phase 3.9):** State persistence, role anchoring, canary capability testing
- ⬜ **30M (Phase 4):** Scaling law validation, emergence verification, blind testing
- ⬜ **125M (Phase 5):** MVP delivery, comprehensive evaluation, roadmap finalization

**Key Metrics for GroundThink:**
- **3.5M:** Proof of life (can overfit, gradients flow, no NaNs)
- **8M:** State persistence (>3 turn memory, role anchoring works), component balance (R/M ratio 0.3-3.0)
- **30M:** Canary pass rate >80% (C1-C5 tests), scaling law R² > 0.95
- **125M:** Production-ready (human eval >4/5), can predict 500M performance within ±10%

**Cross-Reference:** See [CANARY_TESTS.md](CANARY_TESTS.md) for concrete test definitions at each scale.

---

## References

**Strategic Planning:**
- [V4_STRATEGY.md](V4_STRATEGY.md) — Task breakdown by phase
- [VALIDATION_ROADMAP.md](VALIDATION_ROADMAP.md) — 3-week execution plan

**Implementation:**
- [CANARY_TESTS.md](CANARY_TESTS.md) — Concrete behavioral tests
- [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md) — Validation tooling
- [V4_FUSION_MODELS.md](V4_FUSION_MODELS.md) — Variant comparison (A/B testing example)

**Execution:**
- [V4_TRAINING_GUIDE.md](V4_TRAINING_GUIDE.md) — Training procedures
- [V4.5_VALIDATION.md](V4.5_VALIDATION.md) — Validation entries by discovery

---

*Strategic foundation document. All phase-specific details derive from these principles.*  
*Last updated: 2026-01-10*
