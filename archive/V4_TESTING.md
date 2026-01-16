# V4 Testing Strategy

**Created:** 2026-01-08  
**Updated:** 2026-01-09  
**Purpose:** Testing framework for V4 architecture archetypes

---

## 📝 NOTE FOR AGENTS

**When updating this document: Make incremental edits (50-150 lines at a time).**  
Large batch edits cause timeouts and truncation. Work in small sections.

---

## Comprehensive Test Suite

**Automated validation script:** [test_phase0_complete.py](test_phase0_complete.py)

This script runs all gates and tests automatically. Use it after any model changes:
```bash
python test_phase0_complete.py
```

**Output includes:**
- G0-G4 gate results
- Component correctness tests
- Gradient flow verification
- Mini training run (100 steps)
- Component balance analysis
- State evolution check

**Use this before claiming any phase is complete.**

---

## Validation Gates (Quality Gates for All V4+ Development)

**Source:** V3 Section 9.5 (Proven methodology from prior builds)

**Every gate must pass before proceeding to next development phase. No exceptions.**

### Kernel Compatibility Check (G0 - Prerequisites)

**Before running any tests, verify CUDA kernels are available:**

```python
def check_kernel_compatibility():
    """Verify all required kernels are available"""
    kernels = {
        'causal-conv1d': False,
        'selective_scan': False,
        'rwkv6_wkv': False
    }
    
    try:
        import causal_conv1d_cuda
        kernels['causal-conv1d'] = True
    except ImportError:
        print("WARNING: causal-conv1d CUDA kernel not found")
    
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        kernels['selective_scan'] = True
    except ImportError:
        print("WARNING: Mamba selective_scan kernel not found")
    
    try:
        import rwkv6_cuda
        kernels['rwkv6_wkv'] = True
    except ImportError:
        print("WARNING: RWKV-6 CUDA kernel not found")
    
    return kernels

# Usage at start of testing
kernels = check_kernel_compatibility()
if not all(kernels.values()):
    print("⚠ Running with PyTorch fallbacks (slower performance)")
    print("✓ For kernel installation, see V4.5_CUDA_KERNELS.md")
else:
    print("✓ All CUDA kernels available")
```

**Cross-Reference:**
- **Kernel installation:** See [V4.5_CUDA_KERNELS.md](V4.5_CUDA_KERNELS.md) for compilation instructions
- **Troubleshooting:** See [V4.5_CUDA_KERNELS.md - Kernel Compilation Troubleshooting](V4.5_CUDA_KERNELS.md#kernel-compilation-troubleshooting)
- **Performance targets:** See [V4_DIAGNOSTICS.md - Section III: Performance Benchmarking](V4_DIAGNOSTICS.md#iii-performance-benchmarking-suite)

---

### Core Validation Gates (G1-G4)

| Gate | Test | Pass Criteria | Warn Threshold | Fail Condition |
|------|------|---------------|----------------|----------------|
| **G1** | Forward pass | No NaN, correct shapes | - | NaN or shape mismatch |
| **G2** | Init entropy | 2.0-5.0 at step 0 | 6.0-7.0 | <1.0 or >8.0 |
| **G3** | Train 1k steps | Loss decreasing, grad norm 0.5-1.5 | Grad 1.5-3.0 | Grad >5.0 or loss increasing |
| **G3.5** | State health | Cosine <0.99, SVD >0.5, saturation <30% | Cosine 0.95-0.99 | Cosine >0.99 (frozen) |
| **G4** | Component balance | Gradient ratio 0.3-3.0 | 0.1-0.3 or 3-10 | <0.1 or >10 (dead component) |

### Gate Application Schedule

**Apply gates at these checkpoints:**

- **After model build:** Run G1 (forward pass sanity check)
- **Before first training:** Run G1 + G2 (architecture + initialization)
- **After 1K training steps:** Run G3 (training dynamics healthy)
- **Before extended training:** Run G3.5 + G4 (state health + component balance)
- **Before scaling up:** Re-run full gate suite (G1-G4) on larger model

### Gate Check Procedures

**G1 - Forward Pass:**
```python
# Test: Does model forward() work without crashes?
x = torch.randint(0, vocab_size, (batch_size, seq_len))
try:
    logits = model(x)
    assert not torch.isnan(logits).any(), "NaN detected in output"
    assert logits.shape == (batch_size, seq_len, vocab_size), "Shape mismatch"
    print("✓ G1 PASS: Forward pass healthy")
except Exception as e:
    print(f"✗ G1 FAIL: {e}")
```

**G2 - Initialization Entropy:**
```python
# Test: Is model initialized properly (not collapsed or chaotic)?
import torch.nn.functional as F

x = torch.randint(0, vocab_size, (1, 128))
logits = model(x)
probs = F.softmax(logits[0, -1], dim=-1)
entropy = -(probs * torch.log(probs + 1e-9)).sum().item()

if 2.0 <= entropy <= 5.0:
    print(f"✓ G2 PASS: Init entropy = {entropy:.2f}")
elif 6.0 <= entropy <= 7.0:
    print(f"⚠ G2 WARN: Init entropy = {entropy:.2f} (high)")
else:
    print(f"✗ G2 FAIL: Init entropy = {entropy:.2f} (collapsed or chaotic)")
```

**G3 - Training 1K Steps:**
```python
# Test: Does loss decrease with stable gradients?
# Run 1000 training steps, log loss and gradient norms

loss_history = []  # Collect during training
grad_norms = []    # Collect during training

# After 1000 steps:
loss_trend = loss_history[-100:] < loss_history[:100]  # Is it decreasing?
avg_grad_norm = np.mean(grad_norms[-100:])

if loss_trend and 0.5 <= avg_grad_norm <= 1.5:
    print(f"✓ G3 PASS: Loss decreasing, grad norm = {avg_grad_norm:.2f}")
elif 1.5 <= avg_grad_norm <= 3.0:
    print(f"⚠ G3 WARN: Grad norm = {avg_grad_norm:.2f} (high)")
else:
    print(f"✗ G3 FAIL: Loss plateau or grad norm = {avg_grad_norm:.2f}")
```

**G3.5 - State Health:**
```python
# Test: Is hidden state evolving (not frozen)?
# Requires running model on batch and capturing hidden states

from sklearn.metrics.pairwise import cosine_similarity

states = []  # Collect hidden states every 100 steps
for i in range(10):
    state = model.get_hidden_state()  # Capture after forward pass
    states.append(state.detach().cpu().numpy())

# Compute cosine similarity between consecutive states
cosine_sim = cosine_similarity(states[:-1], states[1:]).diagonal().mean()

if cosine_sim < 0.99:
    print(f"✓ G3.5 PASS: State cosine = {cosine_sim:.4f} (evolving)")
elif 0.95 <= cosine_sim <= 0.99:
    print(f"⚠ G3.5 WARN: State cosine = {cosine_sim:.4f} (slow evolution)")
else:
    print(f"✗ G3.5 FAIL: State cosine = {cosine_sim:.4f} (frozen)")
```

**G4 - Component Balance:**
```python
# Test: Are RWKV and Mamba gradients balanced?
# Check gradient ratio between components

rwkv_grads = []
mamba_grads = []

for name, param in model.named_parameters():
    if param.grad is not None:
        if 'rwkv' in name.lower():
            rwkv_grads.append(param.grad.norm().item())
        elif 'mamba' in name.lower():
            mamba_grads.append(param.grad.norm().item())

rwkv_avg = np.mean(rwkv_grads) if rwkv_grads else 0
mamba_avg = np.mean(mamba_grads) if mamba_grads else 0
ratio = rwkv_avg / (mamba_avg + 1e-9)

if 0.3 <= ratio <= 3.0:
    print(f"✓ G4 PASS: Gradient ratio = {ratio:.2f} (balanced)")
elif (0.1 <= ratio < 0.3) or (3.0 < ratio <= 10):
    print(f"⚠ G4 WARN: Gradient ratio = {ratio:.2f} (imbalanced)")
else:
    print(f"✗ G4 FAIL: Gradient ratio = {ratio:.2f} (component dead)")
```

### Stopping Criteria (Additional Quality Checks)

**Stop training immediately if:**
- Val loss increasing >5-10% while train loss decreases (overfitting)
- 2+ LR drops with no improvement (architecture saturated)
- Oscillating loss (up-down >0.5) - indicates architecture conflict
- One component's activations collapse to constant (check G4)
- Gradient ratio <0.1 or >10 (one component is dead)

**Continue training if:**
- Val loss shows "heartbeat" (small dips every few hundred steps)
- Both components have gradient variance (not collapsed)
- Training loss still has tiny downward slope on log scale

---

## Selection Criteria

| Factor | Requirement |
|--------|-------------|
| Parameter Efficiency | Same capabilities at same param count |
| VRAM Footprint | Must fit 6GB with smart offloading |
| Testability | Clear hypothesis for what each excels at |
| Scalability | Path to larger models if successful |

---

## Recommended Testing Trios (Pick One)

### Option A (Conservative)
1. **Dual Pathway** - Tests parallel processing hypothesis
2. **Sequential Specialization** - Tests pipeline hypothesis
3. **Residual State Bridge** - Tests corrective mechanism hypothesis

### Option B (Ambitious)
1. **Hierarchical State Model** - Tests multi-scale state hypothesis
2. **Dynamic Router** - Tests adaptive compute hypothesis
3. **Temporal Scale Separation** - Tests biological plausibility hypothesis

---

## Implementation Notes for 6GB VRAM

```python
# Gradient checkpointing strategy
if archetype in [DualPathway, Hierarchical]:
    checkpoint_every = 2  # More frequent
else:
    checkpoint_every = 4  # Less frequent

# Offloading strategy
cpu_offload_modules = {
    'DualPathway': ['fusion_layers'],
    'Sequential': ['stage_transition'],
    'Hierarchical': ['state_compressor']
}

# Batch size adjustments
batch_size = {
    'DualPathway': 4,
    'Sequential': 6,  # More sequential, less memory
    'Hierarchical': 4
}
```

---

## Required Analysis Metrics

### 1. State Utilization Efficiency
```python
metric = entropy(state_vectors) / max_possible_entropy
# Higher = better state space usage
```

### 2. Cross-Component Gradient Flow
```python
metric = gradient_norm_ratio(RWKV_grads, Mamba_grads)
# Ideal = ~1.0, extreme values indicate imbalance
```

### 3. Information Persistence Score
- **Test:** How long can model maintain specific information?
- **Measure:** Accuracy drop over increasing token distance

### 4. Compute-Performance Curve
- **Measure:** Perplexity improvement per FLOP
- Critical for identifying efficient architectures

### 5. Compute-Efficiency Ratio (CER)

```python
our_cer = (our_perplexity) / (our_compute_flops)
baseline_cer = (baseline_perplexity) / (baseline_compute_flops)

# If our_cer > baseline_cer: Architectural advantage
# If our_cer < baseline_cer: Architecture needs fixes
```

**Purpose:** Compare efficiency against transformer baseline at same parameter count.

### 6. Useful Context Window

| Architecture | Expected Useful Context |
|--------------|-------------------------|
| Standard transformer | 0.25 × trained length |
| GroundThink (claim) | 1.0 × trained length |

**Test Protocol:**
1. Train on 2K context
2. Evaluate at 2K, 4K, 8K, 16K, 32K
3. Measure: Does perplexity degrade gracefully or catastrophically?

### 7. State Persistence Score

**Metric:** How many conversation turns before a key fact is lost?

| Model Type | Expected Turns |
|------------|----------------|
| Transformer baseline | 3-5 turns |
| GroundThink target | 50+ turns |

**Test Protocol:**
1. Inject a key fact at turn 1
2. Query for that fact at turns 5, 10, 20, 50
3. Measure: Accuracy of recall at each distance

---

## Minimum Viable Tests per Archetype

### Synthetic Task Battery (1-2 hours each)
- Copy task (tests state maintenance)
- Selective recall (tests importance filtering)
- Pattern continuation (tests generalization)

### Conversational Flow Test (2-3 hours)
- Multi-turn dialogue consistency
- Topic switching and resumption
- Memory of user preferences

### Efficiency Benchmarks (30 minutes)
- Tokens/second at inference
- Peak memory usage
- Training step time

---

## Testing Progression

### Phase 1: Architecture Validation (Week 1-2)
- Test all 3 selected archetypes at **1B parameter scale**
- Quick elimination of fundamentally flawed designs
- Initial gradient flow and stability checks

### Phase 2: Scale Testing (Week 3-4)
- Take 2 best from Phase 1 to **5.5B parameter scale**
- Full test suite execution
- Efficiency benchmarking
- Identify scaling bottlenecks

### Phase 3: Final Selection (Week 5)
- Take 1 best from Phase 2 to **8B parameter scale**
- Extended training (24-48 hours)
- Comprehensive evaluation
- Decision point: Proceed or iterate

---

## Risk Profile by Archetype

| Risk Level | Archetype | Notes |
|------------|-----------|-------|
| High Risk / High Reward | Dynamic Router | Training instability possible |
| High Risk / High Reward | Temporal Scale Separation | Complex state management |
| Medium Risk | Dual Pathway | Gradient competition issues |
| Medium Risk | Hierarchical State | Compression bottlenecks |
| Lower Risk | Sequential Specialization | Well-understood flow |
| Lower Risk | Residual State Bridge | Conservative approach |

---

## Recommended First Round

**Test these three:**
1. **Sequential Specialization** - Baseline with clear failure modes
2. **Hierarchical State Model** - Tests core "worldview" hypothesis
3. **Residual State Bridge** - Conservative but novel approach

**Why this combination:**
- Covers pipeline, multi-scale, and corrective architectures
- Each has distinct testable hypotheses
- Balanced risk profile (1 low, 1 medium, 1 medium-high)
- Clear decision criteria between them

---

## Critical: Diagnostic Mode

For each archetype, implement diagnostic output:
- Component utilization statistics
- State entropy measurements
- Gradient flow visualizations

---

## Test Suite Metrics for 5-8M Models

### 1. Memory Efficiency Tests

```python
def memory_test(model, batch_size, seq_len):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(torch.randint(0, 10000, (batch_size, seq_len)).cuda())
    return torch.cuda.max_memory_allocated()
```

### 2. Speed Benchmarks

```python
def speed_test(model, seq_len, iterations=100):
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        model(torch.randint(0, 10000, (1, seq_len)).cuda())
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return np.mean(times), np.std(times)
```

### 3. Convergence Tests

```python
def convergence_test(model, train_data, eval_steps=100):
    losses = []
    for step in range(eval_steps):
        loss = training_step(model, train_data)
        losses.append(loss.item())
    
    # Calculate convergence metrics
    initial_loss = np.mean(losses[:10])
    final_loss = np.mean(losses[-10:])
    convergence_rate = (initial_loss - final_loss) / initial_loss
    
    return convergence_rate, losses
```

---

## Critical Experiments for 5M Scale

### Experiment 1: Layer Ratio Sweep

```
Fix total params at 5M
Test RWKV6:Mamba2 ratios:
1. 1:30 (Extreme Mamba)
2. 1:10 (Mamba-heavy)
3. 1:3 (Balanced)
4. 3:1 (RWKV-heavy)
5. 10:1 (Extreme RWKV)
```

### Experiment 2: Hidden Size vs Depth

```
Fix total params at 5M
Test configurations:
1. Hidden=64, Deep (more layers)
2. Hidden=128, Medium
3. Hidden=192, Shallow (fewer layers)
```

### Experiment 3: Fusion Mechanism Comparison

```
Test all 4 fusion mechanisms on same base architecture
Measure: Final loss, training stability, inference speed
```

---

## Implementation Template

```python
class SmallHybridRWKVMamba(nn.Module):
    def __init__(self, vocab_size=10000, hidden_size=128, 
                 rwkv_layers=2, mamba_layers=21, fusion_type='concat'):
        super().__init__()
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # RWKV6 pathway
        self.rwkv_layers = nn.ModuleList([
            RWKV6Layer(hidden_size) for _ in range(rwkv_layers)
        ])
        
        # Mamba2 pathway  
        self.mamba_layers = nn.ModuleList([
            Mamba2Layer(hidden_size) for _ in range(mamba_layers)
        ])
        
        # Fusion
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.fusion = nn.Linear(2 * hidden_size, hidden_size)
        elif fusion_type == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Output
        self.ln_out = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        
        # Tie weights
        self.output.weight = self.embedding.weight
        
    def forward(self, x):
        # Embed
        x_emb = self.embedding(x)
        
        # RWKV pathway
        rwkv_out = x_emb
        for layer in self.rwkv_layers:
            rwkv_out = layer(rwkv_out)
        
        # Mamba pathway
        mamba_out = x_emb
        for layer in self.mamba_layers:
            mamba_out = layer(mamba_out)
        
        # Fuse
        if self.fusion_type == 'concat':
            combined = torch.cat([rwkv_out, mamba_out], dim=-1)
            fused = self.fusion(combined)
        elif self.fusion_type == 'weighted':
            fused = self.alpha * rwkv_out + (1 - self.alpha) * mamba_out
        
        # Output
        fused = self.ln_out(fused)
        logits = self.output(fused)
        
        return logits
```

---

*Testing starts at 5M, not 1B.*
