# Stateful Validation Guide: Testing Grounded Language Models

**Purpose:** Concrete validation framework for testing stateful components (RWKV + Mamba hybrid)

**Context:** Standard LLM validation (loss curves, accuracy) misses stateful dynamics. This guide provides tools to test what makes stateful models different.

**Version:** 4.11-Alpha  
**Created:** 2026-01-10  
**Updated:** 2026-01-10 (Added State Space Fundamentals for 3.5M)  
**Related Docs:** V4.5_VALIDATION.md, VALIDATION_ROADMAP.md, V4_STRATEGY.md Phase 4.0  
**Status:** Implementation reference for Phase 4.0 (Tiny validation) and Phase 3.9 (8M diagnostics)

---

## Part 0: State Space Fundamentals (3.5M — RUN FIRST)

### Why State Monitoring Comes First

Before testing capabilities (memory, grounding, etc.), you must verify the state machinery works:

1. **State vectors must exist** — Architecture produces state outputs
2. **States must initialize properly** — Non-zero, reasonable magnitude  
3. **States must evolve** — Different inputs produce different states
4. **States must be deterministic** — Same input produces same state
5. **Both components must contribute** — RWKV and Mamba both active

**At 3.5M, you cannot test capabilities** (model too small for complex reasoning). But you CAN verify state mechanics.

### State Extraction Requirements

**Current Blocker:** Model's `return_activations` flag returns outputs, not internal states.

**Required API:**
```python
# What we need:
output, states = model(input_ids, return_states=True)

# states should contain:
{
    'rwkv_state': torch.Tensor,  # [batch, layers, hidden_dim] or similar
    'mamba_state': torch.Tensor, # [batch, layers, ssm_state_dim] 
    'gate_values': torch.Tensor, # [batch, seq_len] — fusion gate outputs
}
```

**Implementation needed in [models/hybrid_v4_GF.py](models/hybrid_v4_GF.py):**
```python
def forward(self, x, return_states=False):
    # ... existing forward pass ...
    
    if return_states:
        return output, {
            'rwkv_state': self.rwkv_layer.get_state(),
            'mamba_state': self.mamba_layer.get_state(),
            'gate_values': gate_output
        }
    return output
```

### State Space Test Suite (S0-S4)

**See [CANARY_TESTS.md](CANARY_TESTS.md#s0-s4-state-space-fundamentals-35m-only--required-first) for full test implementations.**

| Test | What It Checks | Pass Criteria |
|------|---------------|---------------|
| S0 | State shapes | Correct dimensions |
| S1 | Initialization | Norm 0.01-100, no NaN |
| S2 | Evolution | Different inputs → different states |
| S3 | Determinism | Same input → same state |
| S4 | Balance | Both components contribute |

### Baseline Results: GF-MH 3.5M (2026-01-10)

**Executed via:** `python tests/test_tiny_graduation.py --states`

| Metric | RWKV | Mamba | Ratio |
|--------|------|-------|-------|
| State shape | [1,4,32] | [1,128] | — |
| State norm | 725.7 | 3.7 | 196x |
| State variance | 9689.4 | 0.089 | 108,583x |
| Evolution diff | 863.2 | 4.5 | 192x |
| Gate value | 0.70 (learned) | — | init was 0.3 |

**Interpretation:**
- All S0-S4 tests pass but S4 shows severe imbalance
- Internal state ratio (108,583x) >> activation ratio (71x from training)
- Mamba's internal state is near-dormant; may act as feedforward layer
- Gate drifted from 0.3→0.7 during training (RWKV dominance increased)

**Graduation Tests (from same session):**
- Task 43 Overfit: Loss 0.48 in 65 steps (10 samples) — healthy learning
- Task 44 Baseline: Val 6.01 vs Random 9.68 — 37.9% improvement

### State Monitoring Metrics

**Capture these during any training run:**
```python
class StateMonitor:
    \"\"\"Track state health during training\"\"\"
    
    def __init__(self):
        self.metrics = {
            'rwkv_norm': [],
            'mamba_norm': [],
            'rwkv_var': [],
            'mamba_var': [],
            'gate_mean': [],
            'activation_ratio': []
        }
    
    def log_step(self, states):
        self.metrics['rwkv_norm'].append(states['rwkv_state'].norm().item())
        self.metrics['mamba_norm'].append(states['mamba_state'].norm().item())
        self.metrics['rwkv_var'].append(states['rwkv_state'].var().item())
        self.metrics['mamba_var'].append(states['mamba_state'].var().item())
        self.metrics['gate_mean'].append(states['gate_values'].mean().item())
        
        ratio = states['rwkv_state'].var() / (states['mamba_state'].var() + 1e-8)
        self.metrics['activation_ratio'].append(ratio.item())
    
    def get_summary(self):
        return {
            'avg_rwkv_var': np.mean(self.metrics['rwkv_var']),
            'avg_mamba_var': np.mean(self.metrics['mamba_var']),
            'avg_activation_ratio': np.mean(self.metrics['activation_ratio']),
            'gate_drift': np.std(self.metrics['gate_mean']),
            'state_stability': 'stable' if np.std(self.metrics['rwkv_norm']) < 1.0 else 'unstable'
        }
```

**Key Insight from Task 40:**
- Activation variance ratio was 71x (RWKV var=8.58, Mamba var=0.12)
- This indicates severe imbalance even with BPE tokenization
- State monitoring during training would have caught this earlier

---

## Part 1: The Stateful Awareness Gap

### The Problem

**Transformers** use attention: "Look back at context when needed"  
**Your Architecture** uses evolving state: "Maintain and update internal memory"

**Why Standard Validation Fails:**
- Loss curves show overall performance
- Accuracy tests show output quality
- But neither shows: "Is state evolving correctly? Does it maintain facts? Can it distinguish users?"

### The Solution: State Tracking Probes

Instead of asking "Is the answer correct?", ask:
1. **Does the state evolve?** (State vectors should change with new information)
2. **Does it remember?** (State should persist facts across turns)
3. **Can it update?** (State should change when corrected)
4. **Can it separate?** (Different contexts should have different states)

---

## Part 2: The 8M Reality Check — What You Probably Missed

### Likely ✅ Validated
- Loss curves decreasing
- Basic accuracy on held-out test set
- Output coherence and fluency
- No NaN/Inf errors during training

### Likely ❌ Not Validated
- **State Contamination:** Does early conversation context leak incorrectly into later unrelated topics?
- **State Persistence Decay:** How many turns before it "forgets" key facts?
- **State Update Consistency:** When you correct the model ("No, actually..."), does it update reliably?
- **Multi-state Tracking:** Can it track two different users' preferences separately?

### Quick Diagnostic Test Suite (Run on Current 8M)

**Test Pattern 1: Short-Term Memory (Baseline)**
```python
dialogue = [
    ("My name is Alex.", "OK, I'll remember that."),
    ("What's my name?", "Your name is Alex.")
]
# Expected: Should pass (1-2 turn memory is easy)
# If fails: Fundamental architecture problem
```

**Test Pattern 2: Medium-Term Memory (Stress)**
```python
dialogue = [
    ("My name is Alex.", "..."),
    ("I prefer coffee over tea.", "..."),
    ("I live in Portland.", "..."),
    ("Dogs are too noisy.", "..."),
    ("My budget is $500.", "..."),
    ("What's my name?", "Your name is Alex."),
    ("What's my budget?", "Your budget is $500.")
]
# Expected: Should pass most questions
# If fails: State decay is too fast
```

**Test Pattern 3: State Correction (Dynamics)**
```python
dialogue = [
    ("I like red.", "OK, you like red."),
    ("Actually, I prefer blue.", "Got it, you prefer blue."),
    ("What's my favorite color?", "Your favorite color is blue.")
]
# Expected: Model updates state, not confused
# If fails: State update mechanism broken
```

**Test Pattern 4: State Separation (Isolation)**
```python
dialogue = [
    ("For work: use formal tone.", "..."),
    ("For personal: use casual tone.", "..."),
    ("Now write a birthday message.", "[Response should be casual]"),
    ("Now write a business proposal.", "[Response should be formal]")
]
# Expected: Model maintains separate style states
# If fails: States bleed into each other
```

---

## Part 3: v4 Validation Suite — Core Components

### Component A: State Tracing Module

**Purpose:** Visualize and analyze state vector evolution during conversations

```python
class StateTracer:
    """Track state evolution throughout conversation"""
    
    def __init__(self, model):
        self.model = model
        self.state_history = []
        self.turn_boundaries = []
    
    def trace_conversation(self, dialogue, return_hidden_states=True):
        """
        Run conversation, capturing state after each turn
        
        Returns:
            - state_vectors: [turn_0_state, turn_1_state, ...]
            - similarities: [similarity(0→1), similarity(1→2), ...]
            - entropy_progression: [entropy_0, entropy_1, ...]
            - abrupt_shifts: [(turn_i, shift_magnitude), ...]
        """
        states = []
        
        for turn_idx, (user_input, context) in enumerate(dialogue):
            # Run forward pass, capture state
            output, hidden_state = self.model(
                user_input, 
                context,
                return_hidden_states=return_hidden_states
            )
            
            states.append({
                'turn': turn_idx,
                'state_vector': hidden_state.detach().cpu().numpy(),
                'state_norm': float(hidden_state.norm()),
                'state_entropy': self._entropy(hidden_state)
            })
        
        # Analyze state trajectories
        similarities = self._compute_turn_similarities(states)
        abrupt_shifts = self._detect_state_shifts(states, threshold=0.5)
        
        return {
            'states': states,
            'similarities': similarities,
            'entropy_progression': [s['state_entropy'] for s in states],
            'abrupt_shifts': abrupt_shifts,
            'total_turns': len(states)
        }
    
    def _compute_turn_similarities(self, states):
        """Cosine similarity between consecutive turn states"""
        sims = []
        for i in range(len(states) - 1):
            sim = self._cosine_sim(states[i]['state_vector'], states[i+1]['state_vector'])
            sims.append({
                'turns': (i, i+1),
                'similarity': sim,
                'change_magnitude': 1 - sim
            })
        return sims
    
    def _detect_state_shifts(self, states, threshold=0.5):
        """Identify abrupt state changes (potential forgetting/confusion)"""
        shifts = []
        for i in range(len(states) - 1):
            change = 1 - self._cosine_sim(states[i]['state_vector'], states[i+1]['state_vector'])
            if change > threshold:
                shifts.append({
                    'between_turns': (i, i+1),
                    'magnitude': change,
                    'flag': 'POTENTIAL_FORGETTING' if change > 0.7 else 'SIGNIFICANT_UPDATE'
                })
        return shifts
    
    def visualize_state_evolution(self, trace_result, save_path='state_evolution.png'):
        """Create visualization dashboard showing state trajectory"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: State norms over time
        norms = [s['state_norm'] for s in trace_result['states']]
        axes[0, 0].plot(norms, marker='o')
        axes[0, 0].set_title('State Norm Evolution')
        axes[0, 0].set_xlabel('Turn')
        axes[0, 0].set_ylabel('||state||')
        
        # Plot 2: Turn similarities (how much state changes)
        sims = [s['similarity'] for s in trace_result['similarities']]
        axes[0, 1].bar(range(len(sims)), sims, color=['red' if s < 0.7 else 'green' for s in sims])
        axes[0, 1].set_title('Turn-to-Turn State Similarity')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].axhline(y=0.7, color='r', linestyle='--', label='Concern Threshold')
        axes[0, 1].legend()
        
        # Plot 3: State entropy
        entropy = trace_result['entropy_progression']
        axes[1, 0].plot(entropy, marker='s')
        axes[1, 0].set_title('State Entropy Over Time')
        axes[1, 0].set_xlabel('Turn')
        axes[1, 0].set_ylabel('Entropy')
        
        # Plot 4: Abrupt shift detection
        shifts = trace_result['abrupt_shifts']
        shift_turns = [s['between_turns'][0] for s in shifts]
        shift_mags = [s['magnitude'] for s in shifts]
        axes[1, 1].scatter(shift_turns, shift_mags, s=100, color='red', alpha=0.6)
        axes[1, 1].set_title('Abrupt State Shifts (Potential Issues)')
        axes[1, 1].set_xlabel('Between Turns')
        axes[1, 1].set_ylabel('Change Magnitude')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        return fig
```

**Metrics Produced:**
- State norm trajectory (should be stable ±10%)
- Turn-to-turn similarity (should stay >0.7 for coherent conversations)
- Abrupt shift detection (red flag if >0.5 magnitude)
- Entropy evolution (should increase with new information)

---

### Component B: Grounding Score Calculator

**Purpose:** Measure how well model stays grounded in conversation context vs. hallucinating

```python
class GroundingScoreCalculator:
    """Evaluate model's grounding in conversation facts"""
    
    def __init__(self, model, context_extractor):
        self.model = model
        self.context_extractor = context_extractor  # NER/semantic parser
    
    def evaluate_response(self, response, context_facts):
        """
        Compute grounding quality of a response
        
        context_facts: {"pet_preference": "cats", "budget": 500, ...}
        response: Model's generated response
        
        Returns:
            - reference_score: How many facts correctly referenced?
            - novelty_score: How much is new info vs. hallucinated?
            - consistency_score: Does it contradict earlier statements?
            - grounding_grade: A/B/C (A=well grounded, C=hallucinating)
        """
        
        # Extract claims from response
        response_claims = self.context_extractor.extract_entities_and_relations(response)
        
        # Compute reference score: correct references / total references
        correct_refs = 0
        total_refs = 0
        for claim in response_claims:
            total_refs += 1
            if self._is_consistent_with_context(claim, context_facts):
                correct_refs += 1
        
        reference_score = correct_refs / max(total_refs, 1)
        
        # Novelty score: new information that doesn't contradict context
        new_claims = [c for c in response_claims if c not in context_facts.values()]
        non_contradictory = sum(1 for c in new_claims if not self._contradicts(c, context_facts))
        novelty_score = non_contradictory / max(len(new_claims), 1)
        
        # Consistency score: no contradictions with context
        contradictions = sum(1 for c in response_claims if self._contradicts(c, context_facts))
        consistency_score = 1.0 - (contradictions / max(total_refs, 1))
        
        # Overall grounding grade
        avg_score = (reference_score + novelty_score + consistency_score) / 3
        if avg_score > 0.85:
            grade = 'A'
        elif avg_score > 0.70:
            grade = 'B'
        else:
            grade = 'C'
        
        return {
            'reference_score': reference_score,      # 0-1
            'novelty_score': novelty_score,          # 0-1
            'consistency_score': consistency_score,  # 0-1
            'grounding_grade': grade,
            'contradictions_found': contradictions,
            'novel_facts': len(new_claims)
        }
    
    def batch_evaluate_dialogue(self, dialogue_history, context_facts):
        """Evaluate entire conversation for grounding drift"""
        scores_over_time = []
        
        for turn_idx, (user_input, response) in enumerate(dialogue_history):
            score = self.evaluate_response(response, context_facts)
            score['turn'] = turn_idx
            scores_over_time.append(score)
        
        # Detect grounding drift
        grades = [s['grounding_grade'] for s in scores_over_time]
        drift_detected = len(set(grades)) > 1  # Grade changes over conversation
        
        return {
            'per_turn_scores': scores_over_time,
            'average_grounding': sum(s['reference_score'] for s in scores_over_time) / len(scores_over_time),
            'drift_detected': drift_detected,
            'recommendation': 'PASS' if not drift_detected else 'REVIEW'
        }
```

**Metrics Produced:**
- **Reference Score:** % of context facts correctly referenced (target: >85%)
- **Novelty Score:** % of new info that doesn't contradict (target: >80%)
- **Consistency Score:** No contradictions with facts (target: >90%)
- **Grounding Grade:** A/B/C (A = well grounded)

---

### Component C: Conversation Genome Analyzer

**Purpose:** Identify patterns in how state evolves across different conversation types

```python
class ConversationGenomeAnalyzer:
    """Analyze state evolution patterns across conversations"""
    
    def __init__(self, model, state_tracer):
        self.model = model
        self.state_tracer = state_tracer
        self.conversation_patterns = {}
    
    def analyze_conversation_type(self, conversation_type, test_dialogues):
        """
        Analyze state patterns for a specific conversation type
        
        conversation_type: "preference_tracking", "multi_user_context", "fact_correction", etc.
        test_dialogues: List of dialogue examples of this type
        
        Returns state evolution pattern
        """
        
        pattern_vectors = []
        pattern_metrics = []
        
        for dialogue in test_dialogues:
            trace = self.state_tracer.trace_conversation(dialogue)
            
            # Extract pattern: [entropy_0, entropy_1, ..., entropy_n]
            pattern = trace['entropy_progression']
            pattern_vectors.append(pattern)
            
            # Compute metrics for this pattern
            metrics = {
                'avg_turn_change': sum(s['change_magnitude'] for s in trace['similarities']) / len(trace['similarities']),
                'max_shift': max(s['change_magnitude'] for s in trace['similarities']),
                'num_abrupt_shifts': len(trace['abrupt_shifts']),
                'final_entropy': trace['entropy_progression'][-1]
            }
            pattern_metrics.append(metrics)
        
        # Identify dominant pattern
        avg_entropy_sequence = self._average_patterns(pattern_vectors)
        
        # Cluster similar conversations
        clusters = self._cluster_conversations(pattern_vectors)
        
        return {
            'conversation_type': conversation_type,
            'num_examples': len(test_dialogues),
            'dominant_pattern': avg_entropy_sequence,
            'pattern_clusters': clusters,
            'average_metrics': self._average_metrics(pattern_metrics),
            'stability': 'stable' if len(clusters) == 1 else 'variable'
        }
    
    def identify_conversation_genomes(self, large_conversation_corpus):
        """
        Cluster all conversations by state evolution pattern
        
        Identifies: "Conversations that trigger pattern X tend to show state behavior Y"
        """
        
        all_patterns = []
        conversation_patterns_map = {}
        
        for conv_id, dialogue in enumerate(large_conversation_corpus):
            trace = self.state_tracer.trace_conversation(dialogue)
            pattern = trace['entropy_progression']
            
            all_patterns.append(pattern)
            conversation_patterns_map[conv_id] = pattern
        
        # Cluster patterns
        genomes = self._cluster_patterns(all_patterns)
        
        # Assign each conversation to a genome
        genome_assignments = {}
        for conv_id, pattern in conversation_patterns_map.items():
            genome = self._nearest_genome(pattern, genomes)
            if genome not in genome_assignments:
                genome_assignments[genome] = []
            genome_assignments[genome].append(conv_id)
        
        return {
            'total_conversations': len(large_conversation_corpus),
            'num_genomes': len(genomes),
            'genomes': genomes,
            'genome_assignments': genome_assignments,
            'genome_frequency': {g: len(convs) for g, convs in genome_assignments.items()}
        }
```

**Patterns Identified:**
- Conversation types that trigger predictable state evolution
- Anomalies (conversations that don't follow expected pattern)
- "Genome health" (are state patterns stable or chaotic?)

---

## Part 4: Validation-First Roadmap for Phase 3.9

### Phase 3.9.1: Baseline Characterization (Week 1-2)

**Goal:** Establish 8M statefulness profile before any scaling decision

**Week 1 Tasks:**
```
1. Run State Tracing Module on 1000 conversation samples
   - Output: State evolution baseline plots
   - Metric: "Average turn-to-turn similarity = X"
   
2. Run Grounding Score Calculator on same conversations
   - Output: Grounding grade distribution
   - Metric: "85% of responses are Grade A"
   
3. Run Conversation Genome Analyzer
   - Output: Identify 3-5 dominant conversation patterns
   - Metric: "Pattern stability = high/medium/low"
```

**Characterization Report Deliverable:**

```markdown
# 8M Statefulness Baseline Profile

## Summary
- Average state persistence: 7.2 turns
- State update reliability: 89%
- Context grounding accuracy: 84%
- Pattern stability: MEDIUM (3 major genomes identified)

## State Persistence Decay Curve
[Plot showing: How long does model remember facts?]
- 1 turn: 98% recall
- 3 turns: 95% recall
- 5 turns: 87% recall
- 8 turns: 71% recall
- 12 turns: 52% recall

## Grounding Grade Distribution
- A (>85% grounded): 68%
- B (70-85% grounded): 24%
- C (<70% grounded): 8%

## Conversation Genomes Discovered
1. **Preference Tracking:** 35% of conversations
   - Stable state evolution
   - High grounding (Grade A: 92%)
   
2. **Multi-Fact Reasoning:** 40% of conversations
   - Variable state evolution
   - Medium grounding (Grade A: 73%)
   
3. **Contradiction Correction:** 25% of conversations
   - Abrupt state shifts detected
   - Lower grounding (Grade A: 61%)

## Critical Issues Found
- State contamination in 12% of "multi-fact" conversations
- State persistence decay faster than ideal (drops to 50% at 12 turns)
- Contradiction handling unreliable (only 71% of corrections stick)

## Recommendations
- Focus Phase 3.9.2 fixes on contradiction handling
- Investigate state persistence mechanisms
- Consider longer training sequences
```

---

### Phase 3.9.2: Architectural Fixes (Week 2-3)

**Based on baseline findings, implement targeted fixes:**

```python
# Example: If state persistence is too fast
class EnhancedStateRetention:
    """Add explicit state reinforcement for low-decay conversations"""
    
    def forward(self, input_ids, hidden_state):
        # Standard forward pass
        output, new_state = self.model(input_ids, hidden_state)
        
        # If model is forgetting too fast, reinforce state
        if self._should_reinforce_state(input_ids, hidden_state):
            # Mix in old state: new_state = α*new_state + (1-α)*old_state
            alpha = 0.7
            reinforced_state = alpha * new_state + (1 - alpha) * hidden_state
            return output, reinforced_state
        
        return output, new_state
```

**Re-run validation after each fix:**
- Run 100 diagnostic dialogues
- Compare metrics to baseline
- Accept fix only if improvement > 5% without regression elsewhere

---

### Phase 3.9.3: Validation-Driven Scaling (Week 3)

**Go/No-Go Criteria for 30M:**

```
✅ GO to 30M if:
  - State vector stability > 0.9 across identical inputs
  - Memory retention > 85% at 8+ turns
  - Grounding accuracy > 85% (Grade A)
  - All conversation genomes show stable patterns
  - Contradiction handling reliability > 90%

❌ NO-GO if:
  - State decay too fast (drops below 70% at 8 turns)
  - State contamination in >10% of conversations
  - Grounding grade drops to C (requires fix)
  - Conversation patterns chaotic (many anomalies)
  - Any genome shows unstable state evolution
```

---

## Part 5: Groundthink-Specific Validation Tests

### Test 1: Linear State Evolution

**Purpose:** Verify state changes are predictable, not chaotic

```python
def test_linear_state_evolution():
    """
    Mamba should show smooth state evolution
    RWKV should show stable attention patterns
    Test: Feed identical context, vary one word
    State vectors should change predictably, not chaotically
    """
    
    base_context = "I prefer coffee over tea. My budget is $500."
    variations = [
        "I prefer coffee over tea. My budget is $500.",        # Identical
        "I prefer tea over coffee. My budget is $500.",         # Change word 1
        "I prefer coffee over tea. My budget is $1000.",        # Change word 2
        "I prefer coffee over tea. My budget is $500 maximum.", # Add words
    ]
    
    states = []
    for variation in variations:
        _, state = model(variation)
        states.append(state.cpu().numpy())
    
    # Analyze state changes
    changes = [np.linalg.norm(states[i+1] - states[0]) for i in range(len(states))]
    
    # Expected: changes scale with modification magnitude
    assert changes[0] < 0.01,      "Identical input should produce identical state"
    assert changes[1] > changes[0], "Single word change should affect state"
    assert changes[2] > changes[1], "Budget change should be larger than word change"
    
    # Linearity check: changes should be proportional
    assert not np.isnan(changes).any(), "State changes should be finite"
    assert max(changes) < 5.0, "State changes shouldn't explode"
    
    return {"test": "PASS", "changes": changes}
```

**Metrics:**
- Predictability score: How well do state changes scale with input modifications?
- Linearity: Are changes proportional or chaotic?

---

### Test 2: Infinite Context Simulation

**Purpose:** Extrapolate beyond training length; test graceful degradation

```python
def test_infinite_context_simulation():
    """
    Train on 1024 tokens, test on longer sequences
    Measure performance decay slope (should be gentle, not cliff)
    """
    
    # Create progressively longer dialogues
    test_lengths = [1024, 2048, 4096, 8192]
    results = {}
    
    for length in test_lengths:
        # Generate dialogue of `length` tokens
        dialogue = generate_long_dialogue(length)
        
        # Run model
        output, final_state = model(dialogue)
        
        # Evaluate: Can it still answer questions about early context?
        question = "What was mentioned in the first message?"
        answer = model.generate(question, context=final_state)
        
        # Score: Is answer correct?
        score = evaluate_answer_correctness(answer, expected="...")
        
        results[length] = {
            'perplexity': compute_perplexity(output, dialogue),
            'context_recall': score,
            'state_norm': final_state.norm().item()
        }
    
    # Analyze decay slope
    ppls = [results[l]['perplexity'] for l in test_lengths]
    decay_rate = compute_slope(test_lengths, ppls)
    
    # Ideal: gentle slope (linear decay is OK, exponential is bad)
    assert decay_rate < 0.005, f"Decay slope {decay_rate} is too steep"
    
    return {"test": "PASS", "decay_rate": decay_rate, "results": results}
```

**Metrics:**
- Decay slope: How much worse does it get with longer context?
- Ceiling: Where does it fail completely?

---

### Test 3: State-Text Alignment

**Purpose:** Verify that internal state matches model's stated knowledge

```python
def test_state_text_alignment():
    """
    After each response, predict: "What is the model's current state?"
    Then: "What would it say next based on that state?"
    Compare prediction vs actual
    """
    
    dialogue = [
        ("My name is Alex and I work in tech.", "OK, noted."),
        ("I have a dog named Buddy.", "Nice!"),
        ("What's my pet's name?", "Your pet is Buddy."),
    ]
    
    for turn_idx, (user_input, expected_response) in enumerate(dialogue):
        # Get actual response
        actual_output, state = model(user_input)
        
        # Predict state: What facts should be in state?
        predicted_state_facts = model.predict_state_contents(state)
        # Expected: {"name": "Alex", "job": "tech", "pet": "Buddy"}
        
        # Predict next: What would model say next given this state?
        query = "Tell me about yourself"
        predicted_next = model.generate(query, context=state)
        
        # Verify: Does predicted_next mention all state facts?
        facts_mentioned = check_facts_in_text(predicted_next, predicted_state_facts)
        
        assert all(facts_mentioned.values()), \
            f"State facts not reflected in generation at turn {turn_idx}"
    
    return {"test": "PASS", "state_text_alignment": "verified"}
```

**Metrics:**
- Alignment score: % of state facts reflected in text
- Consistency: No contradictions between state and output

---

## Part 6: "Graduation to 30M" Checklist

### Technical Validation ✅

- [ ] State vector stability > 0.9 (same input → same state)
- [ ] Memory retention > 85% at 8+ conversation turns
- [ ] State update consistency > 90% (corrections stick)
- [ ] No state contamination detected (<5% of conversations)
- [ ] Grounding accuracy baseline documented (target: >85% Grade A)

### Architectural Validation ✅

- [ ] Loss curves are smooth, no spikes/NaN
- [ ] Gradient flow is smooth through state pathways
- [ ] Parameter scaling: 8M performance > equivalent transformer
- [ ] Handles 8192+ token contexts without degradation cliff
- [ ] State evolution patterns are stable across conversation types

### Capability Validation ✅

- [ ] Linear State Evolution test: PASS
- [ ] Infinite Context Simulation: PASS (graceful degradation)
- [ ] State-Text Alignment: >85% fact consistency
- [ ] Passes 80% of canary diagnostic tasks
- [ ] Shows emergent capabilities in ≥3 domains:
  - [ ] Multi-turn reasoning
  - [ ] Fact correction & update
  - [ ] Multi-context separation

### Resource Validation ✅

- [ ] Training time predictability: ±20% accuracy for 30M estimate
- [ ] VRAM scaling characterized (8M→30M headroom)
- [ ] Bottleneck identified (CPU? GPU memory? Data loading?)
- [ ] 30M validation plan documented (same tools, scaled version)
- [ ] Human eval plan ready (blind comparison vs baseline)

### Final Gate Decision

```
If all sections PASS: ✅ CLEARED FOR 30M SCALING
If any section incomplete: 🟡 EXTEND PHASE 3.9, RE-VALIDATE
If any section FAIL: ❌ RETURN TO ARCHITECTURE, FIX ISSUES
```

---

## Part 7: Implementation Timeline

### Week 1 (Phase 3.9.1: Baseline)
- Day 1-2: Implement State Tracing Module
- Day 3: Implement Grounding Score Calculator
- Day 4: Implement Conversation Genome Analyzer
- Day 5: Run on 1000 conversations, generate baseline report

### Week 2 (Phase 3.9.2: Fixes)
- Day 1-2: Analyze baseline issues
- Day 3-4: Implement fixes (targeted architectural changes)
- Day 5: Re-validate, verify improvements

### Week 3 (Phase 3.9.3: Decision)
- Day 1-2: Run comprehensive final validation suite
- Day 3: Compile go/no-go decision document
- Day 4-5: Plan 30M transition (if GO) or architecture redesign (if NO-GO)

---

## Cross-References

**This guide implements:**
- V4.5_VALIDATION.md → V5 entry (Scaling Strategy) with concrete tools
- VALIDATION_ROADMAP.md → Week 1-3 detailed breakdown with test code
- V4_STRATEGY.md → Phase 3.9 operational framework

**Related documents:**
- V4_DESIGN.md (architecture details for state pathways)
- V4_TRAINING_GUIDE.md (training procedure modifications)
- V4_BUILD_LOG.md (experimental results will be recorded here)

---

*This guide transforms validation from "does it work?" to "how does it think?"*  
*Implementation begins Phase 3.9, Week 1*
