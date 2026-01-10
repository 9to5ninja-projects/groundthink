# Validation Roadmap: 3-Week Plan to 30M Scaling Decision

**Purpose:** Concrete execution plan for validation-first approach (Build → Validate → Fix → Scale)

**Timeline:** Week 1-3 of January 2026  
**Target Decision:** Go/No-Go for 30M scaling by end of Week 3  
**Owner:** GroundThink Team  
**Related Docs:** V4.5_VALIDATION.md (strategic framework), [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md) (implementation code + tests), V4_STRATEGY.md (Phase 4), V4_BUILD_LOG.md (progress tracking)

---

## Executive Summary

| Phase | When | Focus | Gate | Status |
|-------|------|-------|------|--------|
| **Pre-Work** | **Jan 10-12** | Task 40: BPE Baseline | BPE validation complete | ⬜ TODO |
| Baseline | **Week 1** | State Tracing + 4 Diagnostics | Statefulness Report | ⬜ TODO |
| Tooling | **Week 2** | 3 Validation Tools + Baselines | Baseline Metrics | ⬜ TODO |
| Decision | **Week 3** | Fix Issues + Go/No-Go | Validation Gate | ⬜ TODO |

---

## Pre-Work: Task 40 - BPE Benchmark (CRITICAL BLOCKER)

**Objective:** Establish baseline component balance on BPE tokenization BEFORE running diagnostics

**Why This Must Run First:**
- V4.5_VALIDATION.md Entry V4 discovered: Tokenization affects component balance (R/M 0.08-0.11 char-level → 0.20-0.46 BPE)
- Week 1 diagnostics measure state properties; these differ significantly between char-level and BPE
- Week 3 go/no-go decision requires understanding state dynamics at correct tokenization
- All subsequent validation is meaningless if done at wrong tokenization

**Estimated Duration:** 2-3 days (5000+ training steps on FineWeb)

**Command:**
```bash
python train_v4.py \
    --model GF-MH \
    --data fineweb_5m \
    --steps 5000 \
    --output-dir logs/task40_bpe_baseline \
    --checkpoint-interval 500
```

**Success Criteria:**
- Training completes without errors
- Component balance (R/M ratio) matches prediction: 0.20-0.46
- Checkpoint saved at step 5000
- Loss trajectory documented for Week 1 comparison

**Deliverable:** `logs/task40_bpe_baseline/metrics.json` + final checkpoint

**Action:** Start Jan 10, let run overnight. Results ready for Week 1 Day 1.

---

## Week 1: Deploy State Tracing & Run Diagnostics

**Objective:** Understand current 8M model's stateful behavior

### Day 1-2: State Tracing Module Implementation

**What to Build:**
```python
# tools/state_tracing.py
class StateTracer:
    """Capture and visualize RWKV + Mamba hidden state evolution"""
    
    def __init__(self, model):
        self.model = model
        self.traces = {'rwkv': [], 'mamba': []}
    
    def forward_with_trace(self, input_ids, seq_len):
        """Forward pass capturing state at each timestep"""
        # At each decoder layer, hook into:
        # - RWKV hidden state (h, c recurrence)
        # - Mamba SSM state (x_t)
        
        states = self.traces
        return output, states
    
    def plot_evolution(self):
        """Visualize state norm, variance, divergence"""
        # L2 norm over time
        # Dimension-wise analysis
        # Correlation between components
```

**Success Metric:** Can generate plots like:
- RWKV state norm vs step (should be stable ±10%)
- Mamba state norm vs step (same)
- Cross-component correlation (should increase with training)

**Deliverable:** `tools/state_tracing.py` + sample plots

**Estimated Time:** 4-6 hours

---

### Day 3-4: Implement 4 Diagnostic Tests

**Diagnostic Test 1: State Divergence Detection**

```python
def test_state_divergence():
    """Do states grow unboundedly?"""
    
    # Setup: Run model for 1000 steps
    # Measure: L2 norm of RWKV state, Mamba state at steps 1, 100, 500, 1000
    # Expected: Norm stable (ratio <2x between step 1 and 1000)
    # Fail if: Norm explodes (ratio >10x)
    
    results = {
        'rwkv_norm_step1': 1.5,
        'rwkv_norm_step1000': 1.6,  # Ratio 1.07 ✓
        'mamba_norm_step1': 0.8,
        'mamba_norm_step1000': 0.9,  # Ratio 1.12 ✓
        'status': 'PASS'
    }
    return results
```

**Diagnostic Test 2: State Collapse Detection**

```python
def test_state_collapse():
    """Do states freeze/converge prematurely?"""
    
    # Setup: Compare state variance at different training phases (early, mid, late)
    # Measure: Variance of state values within batch, within sequence
    # Expected: Variance increases with training (model learning diversity)
    # Fail if: Variance plateaus (state not learning)
    
    results = {
        'variance_early': 0.1,
        'variance_mid': 0.3,     # ✓ Increasing
        'variance_late': 0.4,    # ✓ Still increasing
        'status': 'PASS'
    }
    return results
```

**Diagnostic Test 3: Component Interaction**

```python
def test_component_interaction():
    """Do both components contribute to state?"""
    
    # Setup: Ablate each component's state (zero it out), measure impact on output
    # Measure: Output change = L2 distance with/without component state
    # Expected: Balanced (both components matter)
    # Fail if: One component >5x more important than other
    
    results = {
        'rwkv_state_impact': 0.05,      # Loss change without RWKV state
        'mamba_state_impact': 0.03,     # Loss change without Mamba state
        'ratio': 1.67,                  # RWKV slightly stronger
        'status': 'PASS'  # Both important, ratio in 0.2-5x range
    }
    return results
```

**Diagnostic Test 4: Long-Range Dependency**

```python
def test_long_range_dependency():
    """Does state maintain useful history at long sequences?"""
    
    # Setup: Run model at seq_len 64, 128, 256
    # Measure: Does perplexity increase linearly or collapse?
    # Expected: Smooth degradation (PPL ratio <2x)
    # Fail if: Catastrophic failure at longer seq (PPL ratio >5x)
    
    results = {
        'ppl_seq64': 5.2,
        'ppl_seq128': 5.8,      # Ratio 1.12 ✓
        'ppl_seq256': 6.5,      # Ratio 1.25 ✓ Smooth degradation
        'status': 'PASS'
    }
    return results
```

**Success Metrics:** All 4 tests pass (PASS status)

**Deliverable:** `eval/diagnostic_tests.py` + results JSON

**Estimated Time:** 6-8 hours

---

### Day 5: Statefulness Report

**What to Document:**
```markdown
# Statefulness Report: 8M Baseline (Week 1)

## Summary
- All 4 diagnostics: PASS
- Model appears healthy for scaling
- State dynamics as expected for hybrid

## Detailed Findings

### D1: State Divergence
- RWKV norm stable: 1.5 → 1.6 (ratio 1.07) ✓
- Mamba norm stable: 0.8 → 0.9 (ratio 1.12) ✓

### D2: State Collapse
- Variance increases throughout training ✓
- No evidence of premature freezing

### D3: Component Interaction  
- Both components contribute
- RWKV slightly stronger (ratio 1.67, target 0.2-5.0) ✓

### D4: Long-Range Dependency
- Smooth degradation with seq length ✓
- No catastrophic failures at 256 tokens

## Recommendations
- Proceed to Week 2 (Tool implementation)
- No architectural fixes needed at baseline
- Monitor these metrics during extended training
```

**Deliverable:** `reports/statefulness_report_week1.md` + visualizations

**Estimated Time:** 2-4 hours

---

## Week 2: Build Validation Tools & Establish Baselines

**Objective:** Create reproducible validation infrastructure and define "good enough" thresholds

### Day 1-2: State Health Monitor

```python
# tools/state_health_monitor.py
class StateHealthMonitor:
    """Track state norm evolution during training"""
    
    def __init__(self, model):
        self.model = model
        self.history = {'step': [], 'rwkv_norm': [], 'mamba_norm': []}
    
    def log_state(self, step, rwkv_state, mamba_state):
        """Called after each training step"""
        self.history['step'].append(step)
        self.history['rwkv_norm'].append(rwkv_state.norm().item())
        self.history['mamba_norm'].append(mamba_state.norm().item())
    
    def is_healthy(self, threshold_deviation=0.1):
        """Check if health metric within expected range"""
        # Compute moving average of norms
        # Check for divergence (norm growing >threshold)
        # Return: True/False + metric value
        ...
```

**Success Metric:** Monitor operational during 100-step training run, produces plots

**Integration:** Add to train_v4.py logging

**Estimated Time:** 4-6 hours

---

### Day 2-3: Gradient-State Coupling Analyzer

```python
# tools/gradient_coupling_analyzer.py
class CouplingAnalyzer:
    """Measure correlation between state gradients and loss"""
    
    def __init__(self, model):
        self.model = model
    
    def compute_coupling(self, loss, rwkv_state, mamba_state):
        """
        For each component's state:
        - Compute gradient w.r.t. loss
        - Compute gradient w.r.t. state
        - Return correlation (0=independent, 1=perfect)
        """
        # Correlation between:
        # d(loss)/d(RWKV state) and actual state values
        # d(loss)/d(Mamba state) and actual state values
        ...
    
    def component_importance(self):
        """Which component's state affects loss more?"""
        # Return coupling strength for each component
        # Balanced: both ~0.5
        # Imbalanced: one >0.8, other <0.2
```

**Success Metric:** Produces coupling scores, identifies component importance

**Threshold Definition:**
- Balanced coupling: both components 0.4-0.6
- Acceptable imbalance: ratio <2.5x
- Red flag: one component <0.2 (basically unused)

**Estimated Time:** 5-7 hours

---

### Day 4-5: Information Flow Tracer

```python
# tools/information_flow_tracer.py
class InformationFlowTracer:
    """Measure mutual information between states and outputs"""
    
    def __init__(self, model):
        self.model = model
    
    def trace_information_flow(self, input_ids, seq_len):
        """
        For each component:
        - Compute: MI(state_t-1 → output_t)
        - This tells us: How much of output depends on state?
        
        High MI = state actively used in prediction
        Low MI = state basically ignored
        """
        rwkv_mi = self.mutual_information(rwkv_state_prev, output_curr)
        mamba_mi = self.mutual_information(mamba_state_prev, output_curr)
        
        return {'rwkv_mi': rwkv_mi, 'mamba_mi': mamba_mi}
    
    def is_both_components_active(self, threshold=0.2):
        """Both components should contribute >threshold MI"""
        rwkv_mi = self.trace_information_flow(...)['rwkv_mi']
        mamba_mi = self.trace_information_flow(...)['mamba_mi']
        
        return rwkv_mi > threshold and mamba_mi > threshold
```

**Success Metric:** Produces information flow metrics, identifies bottlenecks

**Threshold Definition:**
- Both components active: each >20% of total information flow
- Red flag: one component <5% (essentially dead)

**Estimated Time:** 6-8 hours

---

### Day 5: Baseline Establishment

**Run Current 8M Model with All Tools Active**

```bash
# Train for 5000 steps with full instrumentation
python train_v4.py \
    --model GF-MH \
    --steps 5000 \
    --enable-state-health-monitor \
    --enable-coupling-analyzer \
    --enable-information-flow-tracer \
    --output-dir logs/baseline_8m_instrumented
```

**Metrics to Record:**

```json
{
  "state_health": {
    "rwkv_norm_stability": 0.12,      // Deviation from mean (target: <0.15)
    "mamba_norm_stability": 0.11,      // (target: <0.15)
    "status": "PASS"
  },
  "gradient_coupling": {
    "rwkv_coupling_strength": 0.52,    // (target: 0.4-0.6)
    "mamba_coupling_strength": 0.48,   // (target: 0.4-0.6)
    "ratio": 1.08,                     // (target: <2.5)
    "status": "PASS"
  },
  "information_flow": {
    "rwkv_mi_fraction": 0.48,          // (target: >0.2)
    "mamba_mi_fraction": 0.52,         // (target: >0.2)
    "ratio": 1.08,                     // (target: 0.2-5.0)
    "status": "PASS"
  },
  "overall_status": "PASS"
}
```

**Deliverable:** `metrics/baseline_8m_metrics.json` + comprehensive report

**Estimated Time:** 2-4 hours + 1-2 hours training time

---

### Baseline Documentation

Create `BASELINE_METRICS.md`:
```markdown
# 8M Model Baseline Metrics (Week 2)

## Threshold Definitions

| Metric | Low | OK | High | Status |
|--------|-----|----|----|--------|
| State health deviation | >0.2 | 0.1-0.2 | <0.1 | OK |
| Component coupling ratio | >3.0 | 1.5-3.0 | <1.5 | OK |
| MI min (either component) | <0.1 | 0.1-0.2 | >0.2 | OK |

## Current Baseline

All metrics in OK range. Model appears healthy for scaling.

## "Good Enough" for 8M Graduation

Model can proceed to 30M if all metrics remain in OK range through Week 3.
```

---

## Week 3: Fix Issues & Make Go/No-Go Decision

**Conditional Paths Based on Week 1-2 Results**

### Path A: All Metrics Healthy (Expected)

**Day 1-2: Extended Validation**

```bash
# Run on BPE data (Task 40) with all tools active
python train_v4.py \
    --model GF-MH \
    --data fineweb_bpe \
    --steps 10000 \
    --enable-all-monitoring \
    --output-dir logs/week3_extended
```

**Verification Checklist:**
- [ ] Metrics stable through 10K steps
- [ ] No degradation vs Week 2 baseline
- [ ] State health remains in OK range
- [ ] Component coupling balanced
- [ ] Information flow shows both components active

**Day 3: Go/No-Go Decision**

```markdown
# VALIDATION GATE: PASS ✅

## Metrics Summary
- State Health: PASS (deviation 0.11-0.12)
- Component Coupling: PASS (ratio 1.08)
- Information Flow: PASS (both >0.45)
- Extended Training: PASS (stable through 10K)
- BPE Validation: PASS (R/M ratio 0.25)

## Decision: GO TO 30M ✅

**Reasoning:** All validation criteria met. Model demonstrates:
1. Stable, non-diverging state evolution
2. Balanced component utilization
3. Information flow through both components
4. No architectural weaknesses detected

**Next Step:** Begin 30M model design (Phase 4)
```

---

### Path B: Issues Detected (Unlikely, but plan for it)

**Day 1: Root Cause Analysis**

If metrics show:
- **State divergence** → Check initialization, add gradient clipping
- **Component uncoupling** → Adjust mamba_lr_mult, add coupling loss
- **Information bottleneck** → Expand hidden dim, check architecture

**Day 2: Implement Fix**

```python
# Example: If state divergence detected
# Add gradient clipping to RWKV/Mamba states

class StateGradientClipper:
    def clip_gradients(self, rwkv_state, mamba_state, max_norm=1.0):
        rwkv_state.grad.clip_(max=max_norm)
        mamba_state.grad.clip_(max=max_norm)
```

**Day 3: Re-validate**

Run 5K steps with fix, verify metrics improve to baseline.

**Decision:**

```markdown
# VALIDATION GATE: FIX & REVALIDATE

## Issue Found & Fixed
- Issue: State divergence detected
- Fix: Added gradient clipping (max_norm=1.0)
- Result: Metrics now in OK range

## New Baseline After Fix
[metrics after fix]

## Decision: GO TO 30M (FIXED) ✅

**Next Step:** Begin 30M with gradient clipping enabled
```

---

## Deliverables Summary

### Week 1 Outputs
- [ ] `tools/state_tracing.py` — State visualization module
- [ ] `eval/diagnostic_tests.py` — 4 diagnostic tests
- [ ] `reports/statefulness_report_week1.md` — Findings + plots
- [ ] Sample visualizations (state norm vs step, etc.)

### Week 2 Outputs
- [ ] `tools/state_health_monitor.py` — Production monitoring
- [ ] `tools/gradient_coupling_analyzer.py` — Coupling analysis
- [ ] `tools/information_flow_tracer.py` — Information flow measurement
- [ ] `metrics/baseline_8m_metrics.json` — Baseline numbers
- [ ] `BASELINE_METRICS.md` — Threshold definitions + interpretation

### Week 3 Outputs
- [ ] `logs/week3_extended/` — Extended training results
- [ ] `VALIDATION_GATE_PASS.md` or `VALIDATION_GATE_FAIL.md` — Go/No-Go decision
- [ ] Updated architecture (if fixes needed)
- [ ] Recommendation for Phase 4 (30M scaling)

---

## Success Criteria (Final Check)

✅ **All outputs completed on schedule**
✅ **All tools operational and integrated**
✅ **Baseline metrics documented**
✅ **Go/No-Go decision made with confidence**
✅ **Clear path forward for Phase 4**

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Tools too slow to implement | Use simpler versions first, iterate |
| Week 1 diagnostics fail | Issue likely design bug; good to catch now |
| Metrics unclear/contradictory | Add manual inspection of state tensors |
| Week 3 inconclusive | Extend validation by 1 week, don't force decision |
| Tools conflict with training | Implement as optional hooks, can disable |

---

*Validation-first approach: Invest 3 weeks now to save months of debugging at 30M scale.*

*Last updated: 2026-01-10*
