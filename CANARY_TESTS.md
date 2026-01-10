# Canary Tests: Architecture-Specific Evaluation Suite

**Purpose:** Concrete behavioral tests tuned for state-based hybrid models (RWKV-6 + Mamba-2). Replace generic benchmarks with tests that reveal statefulness.

**Version:** 4.10-Alpha  
**Created:** 2026-01-10  
**Related Docs:** V4.5_VALIDATION.md (V7 entry), STATEFUL_VALIDATION_GUIDE.md, V4_TRAINING_GUIDE.md

---

## Canary Test Framework

**Goal:** Answer "Can this model maintain and use internal state across extended conversations?"

**Standard Baseline:** Transformer baseline (e.g., GPT-2) fails most of these with >10 turns.

---

## S0-S4: State Space Fundamentals (3.5M ONLY — REQUIRED FIRST)

**Purpose:** Before testing capabilities, verify the state machinery works. These are the foundation tests.

### S0: State Vector Shape Verification
```python
def test_state_shapes():
    """Verify state vectors exist and have correct dimensions"""
    model = load_model('GF-MH')
    x = torch.randint(0, vocab_size, (1, 32))
    
    output, states = model(x, return_states=True)
    
    # Check RWKV state exists
    assert 'rwkv_state' in states, "RWKV state missing"
    assert states['rwkv_state'].shape[-1] == hidden_dim, "RWKV state dim mismatch"
    
    # Check Mamba state exists  
    assert 'mamba_state' in states, "Mamba state missing"
    assert states['mamba_state'].shape[-1] == hidden_dim, "Mamba state dim mismatch"
    
    print(f"✓ S0 PASS: RWKV state {states['rwkv_state'].shape}, Mamba state {states['mamba_state'].shape}")

Difficulty: Trivial (infrastructure check)
Success Criterion: Must pass 100%
Failure: Architecture broken — do not proceed
```

### S1: State Initialization Health
```python
def test_state_initialization():
    """Verify states initialize to reasonable values"""
    model = load_model('GF-MH')
    x = torch.randint(0, vocab_size, (1, 32))
    
    output, states = model(x, return_states=True)
    
    rwkv_norm = states['rwkv_state'].norm().item()
    mamba_norm = states['mamba_state'].norm().item()
    
    # States should be non-zero and not exploded
    assert 0.01 < rwkv_norm < 100, f"RWKV state norm {rwkv_norm} out of range"
    assert 0.01 < mamba_norm < 100, f"Mamba state norm {mamba_norm} out of range"
    
    # Check for NaN/Inf
    assert not torch.isnan(states['rwkv_state']).any(), "RWKV state has NaN"
    assert not torch.isnan(states['mamba_state']).any(), "Mamba state has NaN"
    
    print(f"✓ S1 PASS: RWKV norm={rwkv_norm:.3f}, Mamba norm={mamba_norm:.3f}")

Difficulty: Trivial
Success Criterion: Must pass 100%  
Failure: Initialization broken — check weight init
```

### S2: State Evolution (Different Inputs → Different States)
```python
def test_state_evolution():
    """Verify states change with different inputs"""
    model = load_model('GF-MH')
    
    input_a = tokenize("The cat sat on the mat.")
    input_b = tokenize("Machine learning is fascinating.")
    
    _, states_a = model(input_a, return_states=True)
    _, states_b = model(input_b, return_states=True)
    
    # States should differ for different inputs
    rwkv_diff = (states_a['rwkv_state'] - states_b['rwkv_state']).norm().item()
    mamba_diff = (states_a['mamba_state'] - states_b['mamba_state']).norm().item()
    
    assert rwkv_diff > 0.1, f"RWKV states too similar: diff={rwkv_diff}"
    assert mamba_diff > 0.1, f"Mamba states too similar: diff={mamba_diff}"
    
    print(f"✓ S2 PASS: States differ (RWKV diff={rwkv_diff:.3f}, Mamba diff={mamba_diff:.3f})")

Difficulty: Easy
Success Criterion: Must pass 100%
Failure: State not encoding input — check forward pass
```

### S3: State Determinism (Same Input → Same State)
```python
def test_state_determinism():
    """Verify same input produces same state (no random noise)"""
    model = load_model('GF-MH')
    model.eval()  # Disable dropout
    
    input_x = tokenize("Hello world, this is a test.")
    
    with torch.no_grad():
        _, states_1 = model(input_x, return_states=True)
        _, states_2 = model(input_x, return_states=True)
    
    rwkv_diff = (states_1['rwkv_state'] - states_2['rwkv_state']).norm().item()
    mamba_diff = (states_1['mamba_state'] - states_2['mamba_state']).norm().item()
    
    assert rwkv_diff < 1e-5, f"RWKV states non-deterministic: diff={rwkv_diff}"
    assert mamba_diff < 1e-5, f"Mamba states non-deterministic: diff={mamba_diff}"
    
    print(f"✓ S3 PASS: States deterministic (diff < 1e-5)")

Difficulty: Easy
Success Criterion: Must pass 100%
Failure: Non-deterministic forward pass — check dropout/randomness
```

### S4: Component Contribution (Both Components Active)
```python
def test_component_contribution():
    """Verify both RWKV and Mamba contribute to output"""
    model = load_model('GF-MH')
    x = tokenize("The quick brown fox jumps over the lazy dog.")
    
    # Get state variances (proxy for activity)
    _, states = model(x, return_states=True)
    
    rwkv_var = states['rwkv_state'].var().item()
    mamba_var = states['mamba_state'].var().item()
    
    # Variance ratio (1.0 = perfectly balanced)
    ratio = rwkv_var / (mamba_var + 1e-8)
    
    # WARN if severely imbalanced, but don't FAIL at Tiny scale
    if ratio > 100 or ratio < 0.01:
        print(f"⚠ S4 WARN: Severe imbalance (ratio={ratio:.2f})")
        print(f"  RWKV var={rwkv_var:.6f}, Mamba var={mamba_var:.6f}")
    else:
        print(f"✓ S4 PASS: Both components active (ratio={ratio:.2f})")
    
    return {'rwkv_var': rwkv_var, 'mamba_var': mamba_var, 'ratio': ratio}

Difficulty: Medium (may reveal architectural issues)
Success Criterion: ratio between 0.01 and 100 (WARN if outside 0.1-10)
Failure: One component completely dead — investigate architecture
```

**State Space Metrics Summary:**
| Test | Pass | Warn | Fail |
|------|------|------|------|
| S0 | Shapes correct | — | Wrong shapes |
| S1 | Norm 0.01-100 | — | Norm outside range or NaN |
| S2 | Diff > 0.1 | — | States identical |
| S3 | Diff < 1e-5 | — | Non-deterministic |
| S4 | Ratio 0.01-100 | Ratio 0.1-10 | Component dead |

---

## C1: State Persistence (Baseline)

**Tests:** Can model remember a stated fact across intervening context?

### C1a: 1-Turn Recall (Control)
```
Input:
  "My favorite color is blue. What is my favorite color?"

Expected Output:
  "Your favorite color is blue."

Difficulty: Easy (immediate)
Success Criterion: Must pass (>95%)
Failure: Architectural problem
```

### C1b: 5-Intervening Sentences
```
Input:
  Turn 1: "My favorite color is blue."
  Turn 2: "Yesterday I went to the park."
  Turn 3: "I saw some birds flying overhead."
  Turn 4: "The weather was nice and warm."
  Turn 5: "I enjoyed my walk in the fresh air."
  Turn 6: "What is my favorite color?"

Expected Output:
  "Your favorite color is blue."

Difficulty: Medium
Success Criterion: >85%
Failure: State decay too fast
```

### C1c: 10-Intervening Sentences
```
Input:
  [1 statement + 10 unrelated sentences + 1 query]

Expected Output:
  Correct recall

Difficulty: Hard
Success Criterion: >70%
Failure: State degrades significantly
```

### C1d: Multi-Fact Recall
```
Input:
  Turn 1: "My favorite color is blue and my name is Alex."
  Turn 2-6: [5 unrelated sentences]
  Turn 7: "What are my name and favorite color?"

Expected Output:
  Mentions both "Alex" and "blue"

Difficulty: Medium-Hard
Success Criterion: >80% (must recall both facts)
Failure: State only partially encodes information
```

**Metric:** State Persistence Score = (turns_before_forgetting / max_expected_turns) × 100

---

## C2: Long-Context Grounding (Distributed Facts)

**Tests:** Can model synthesize facts from distant positions in context?

### C2a: Three Facts at Varying Depths
```
Context (2000 tokens):
  - Position 100: "My budget is $500."
  - Position 1500: "I need WiFi connectivity."
  - Position 1950: "I prefer fast delivery (2 days)."

Query:
  "Based on everything I said, would you recommend a $600 laptop 
   without WiFi for me?"

Expected Output:
  "No, because you have a $500 budget (exceeds it) and need WiFi."

Difficulty: Hard
Success Criterion: >75% (uses multiple distant facts)
Failure: Attends only to recent context
```

### C2b: Implicit Synthesis
```
Context:
  [Facts scattered across 2000 tokens]

Query:
  "What should I avoid based on what I mentioned?"

Expected Output:
  Synthesizes constraints from multiple positions

Difficulty: Very Hard
Success Criterion: >60%
Failure: Model misses implicit connections
```

**Metric:** Context Grounding Score = (questions_answered_correctly / total_questions) × 100

---

## C3: Conversational State Tracking (Multi-Turn Dialogue)

**Tests:** Does state evolve correctly across many turns without collapse or contamination?

### C3a: Simple Preference Tracking
```
Turn 1: (System greeting)
Turn 2: "I prefer cats over dogs."
Turn 3: (Unrelated topic)
Turn 4: (Unrelated topic)
Turn 5: (Unrelated topic)
Turn 6: (Unrelated topic)
Turn 7: "What pet do I prefer?"

Expected: "You prefer cats."

Difficulty: Medium
Success Criterion: >90%
Failure: State lost or overwritten
```

### C3b: Implicit Reference Resolution
```
Turn 3: "I prefer cats because they're independent."
Turn 4-6: (Other topics)
Turn 7: "Why do I prefer that?"

Expected: References "independence" or "cats"

Difficulty: Hard
Success Criterion: >80% (correctly resolves "that")
Failure: Model doesn't maintain referential state
```

### C3c: State Correction & Update
```
Turn 1: "I like red."
Turn 2: "Actually, I prefer blue."
Turn 3-5: (Unrelated)
Turn 6: "What's my favorite color?"

Expected: "Your favorite color is blue."

Difficulty: Medium
Success Criterion: >85% (uses most recent state, not first)
Failure: State update mechanism broken
```

**Metric:** Dialogue State Accuracy = (turns_with_correct_state / total_turns) × 100

---

## C4: Instruction Following with State Persistence

**Tests:** Can model maintain behavioral instructions across multiple turns?

### C4a: Format Persistence
```
Turn 1: "Always format lists with bullet points."
Turn 2: (Unrelated)
Turn 3: "List three fruits."

Expected Output:
  • Apple
  • Banana
  • Orange

Turn 4: (Unrelated)
Turn 5: "List three vegetables."

Expected Output:
  • Carrot
  • Broccoli
  • Spinach

Difficulty: Medium
Success Criterion: >90% (format consistency across both lists)
Failure: Format instruction lost
```

### C4b: Tone/Style Persistence
```
Turn 1: "Respond in a formal, professional tone."
Turn 2: (Unrelated)
Turn 3: "Tell me about coffee."

Expected: Formal, professional response

Turn 4: (Unrelated)
Turn 5: "What about tea?"

Expected: Still formal, professional

Difficulty: Hard
Success Criterion: >80% (tone consistency >90% across turns)
Failure: State instruction doesn't persist
```

### C4c: Complex Multi-Instruction Composition
```
Turn 1: "Be formal AND concise AND use technical terms."
Turn 2-4: (Unrelated, multi-turn)
Turn 5: "Explain machine learning."

Expected: Formal + concise + technical vocabulary

Difficulty: Very Hard
Success Criterion: >70% (all 3 instructions maintained)
Failure: Only partial instruction compliance
```

**Metric:** Instruction Persistence = (turns_maintaining_instructions / total_instructed_turns) × 100

---

## C5: Role/Persona Consistency

**Tests:** Does model develop and maintain a coherent persona?

### C5a: Persona Emergence (Turn 1)
```
Turn 1: "I'm a software engineer interested in AI."
Turn 2-3: (Exploratory questions)
Turn 4: (Unrelated topic A)
Turn 5: (Unrelated topic B)
Turn 6: "Tell me about your interests."

Expected: References "software engineering" or "AI"

Difficulty: Medium
Success Criterion: >85%
```

### C5b: Persona Consistency (5+ Turns)
```
[5+ turn dialogue establishing persona]

Measure: Tone, vocabulary, reference consistency

Difficulty: Hard
Success Criterion: >80% consistency score
Failure: Persona drifts or contradicts itself
```

### C5c: Persona Under Contradiction
```
Turn 1: "I'm very organized and detail-oriented."
Turn 5: "Actually, I'm pretty chaotic."
Turn 10: "How would you describe me?"

Expected: Model either acknowledges the contradiction ("You mentioned being both organized and chaotic") OR adopts the latest statement consistently ("You describe yourself as chaotic"). Must not revert to contradicted claim.

Difficulty: Hard
Success Criterion: >75% (handles contradiction gracefully, does not revert)
Failure: Confusion or inconsistent persona
```

**Metric:** Persona Consistency = (turns_maintaining_coherent_persona / total_turns) × 100

---

## C6: State Bleeding Detection (Isolation Test)

**Tests:** Do separate conversations maintain distinct states?

### C6a: Conversation A → B → A
```
Conversation A:
  Turn 1: "My favorite food is pizza."
  Turn 2: "What do I like?"

Expected: "You like pizza."

[Switch to Conversation B]
Conversation B:
  Turn 1: "My favorite food is sushi."
  Turn 2: "What do I like?"

Expected: "You like sushi."

[Switch back to Conversation A]
Conversation A:
  Turn 3: "Remind me what I like."

Expected: "You like pizza." (not sushi)

Difficulty: Hard
Success Criterion: >90% (must isolate conversations)
Failure: State bleeds between conversations
```

### C6b: Multi-User Context
```
Context Setup:
  User A: "I'm a teacher."
  User B: "I'm an engineer."

Query A: "What's your job?"
Expected: "I'm a teacher."

Query B: "What's your job?"
Expected: "I'm an engineer."

Difficulty: Very Hard
Success Criterion: >85%
Failure: Model confuses user identities
```

**Metric:** State Isolation Score = (conversations_properly_isolated / total_tests) × 100

---

## Test Suite Composition

### Tiny Suite (3.5M — State Space Fundamentals)

**Purpose:** Verify state spaces are functional before any capability testing. These are infrastructure tests, not capability tests.

**State Space Monitoring (Required First):**
- S0: State vector shape verification (RWKV + Mamba states exist and have correct dimensions)
- S1: State initialization (states are non-zero, reasonable magnitude)
- S2: State evolution (states change with different inputs)
- S3: State determinism (same input → same state)
- S4: Component contribution (both RWKV and Mamba states are non-trivial)

**Minimal Canary (After State Verification):**
- C1a (1-turn baseline) — Must pass >95%
- C1b-lite (3-intervening, not 5) — Adjusted for Tiny capacity

**Time:** ~5 minutes  
**Purpose:** Verify training infrastructure and state mechanics work before 8M scaling

**CRITICAL:** Do NOT run capability tests (C2-C6) on Tiny models. They lack capacity for complex reasoning. Focus on state space health.

### Minimal Suite (8M+ Quick Validation)
- C1a (1-turn baseline)
- C1b (5-intervening)
- C4a (format persistence)
- C5a (persona emergence)

**Time:** ~2 minutes per scale  
**Purpose:** Smoke test before extended training

### Standard Suite (Phase 3.9 Week 1)
- C1a, C1b, C1c (persistence progression)
- C2a (3-fact grounding)
- C3a, C3b (simple + implicit tracking)
- C4a, C4b (format + tone)
- C5a, C5b (persona)
- C6a (conversation isolation)

**Time:** ~20 minutes per scale  
**Purpose:** Comprehensive baseline before Phase 4

### Extended Suite (Phase 3.9 Week 2-3)
- All tests above PLUS:
- C1d (multi-fact recall)
- C2b (implicit synthesis)
- C3c (state correction)
- C4c (complex instructions)
- C5c (contradiction handling)
- C6b (multi-user isolation)

**Time:** ~45 minutes per scale  
**Purpose:** Deep validation before 30M decision

---

## Scoring Rubric

### Per-Test Score (0-100)
- 90-100: ✅ PASS (meets canary expectation)
- 70-89: 🟡 CONCERN (degraded but functional)
- 50-69: ⚠️ WARNING (significant issues)
- <50: ❌ FAIL (architectural problem)

### Overall Canary Score
```
Minimal Suite Average ≥85%: Safe to proceed to standard training
Standard Suite Average ≥80%: Safe to proceed to 30M scaling
Extended Suite Average ≥75%: Safe to proceed to 125M+ scaling
Any category <50%: STOP - investigate architectural issue
```

### Scaling Thresholds
| Scale | State Tests | Minimal | Standard | Extended |
|-------|-------------|---------|----------|----------|
| 3.5M | S0-S4: 100% | 85% (C1a, C1b-lite) | N/A | N/A |
| 8M | S0-S4: 100% | 90% | 80% | N/A |
| 30M | S0-S4: 100% | 90% | 85% | 75% |
| 125M | S0-S4: 100% | 90% | 90% | 85% |

**Note:** State tests (S0-S4) must pass 100% at ALL scales. They are infrastructure tests, not capability tests.

---

## Implementation Notes

### Automation
- Each canary should be runnable in ~30 seconds
- Collect outputs in JSON for tracking across scales
- Build dashboard showing pass/fail per test, per scale

### Bias Mitigation
- Run 3-5 dialogue variants per test (vary vocabulary, phrasing)
- Average results across variants
- Flag outliers (single-run failures)

### Baseline Comparison
- Run minimal suite on GPT-2 (for reference)
- Expected: 40-50% average (shows these are harder than standard benchmarks)
- Your hybrid should exceed GPT-2 at all scales

---

## Canary Test Roadmap

### Phase 3.9, Week 1
- Implement minimal suite on 8M
- Establish baseline metrics
- Document any failures

### Phase 3.9, Week 2-3
- Run standard suite
- Implement fixes if needed
- Decision: Ready for 30M?

### Phase 4, Week 1
- Run standard suite on 30M
- Compare to 8M baselines

### Phase 4, Week 2-3
- Run extended suite
- Compile final "Canary Report"
- Publish if results are strong

---

*This framework replaces generic benchmarks with tests specifically designed for state-based architectures. Success means your hybrid remembers, synthesizes, and maintains context like a real conversation partner.*
