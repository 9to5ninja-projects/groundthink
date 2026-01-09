# V4 Testing Strategy

**Created:** 2026-01-08  
**Purpose:** Testing framework for V4 architecture archetypes

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
