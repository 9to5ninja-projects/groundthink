# V4 Diagnostics & Benchmarks

**Created:** 2026-01-08  
**Purpose:** Architecture-agnostic diagnostics targeting stateful model capabilities

---

## Context

Traditional benchmarks don't capture stateful model strengths. We need diagnostics that:
- Are architecture-agnostic
- Target specific capabilities we expect
- Compare fairly against baselines (pure RWKV, pure Mamba, transformer)

---

## 1. State Dynamics Diagnostics

### State Entropy Over Time
- Measure how state information content changes during processing
- **Expectation:** Important info preserved, unimportant info decays

### State Update Magnitude Distribution
- Track norms of state updates
- Consistent or spiky? Spikes = unstable training or important events

### State Similarity Across Sequences
- Similar inputs → similar states?
- Different inputs → divergent states?
- Tests if state captures meaningful information

---

## 2. Forgetting and Remembering Tests

### Long-Range Dependency Tasks
Not just copying - tasks requiring holding and using information over long intervals.

**Example:**
> "The number X is ... (1000 tokens of filler) ... What was X?"

### Importance-Based Forgetting
Test if model forgets irrelevant information.

**Example:** Provide story with many details, ask about central vs peripheral details.

---

## 3. Robustness Tests

### Noise Injection
- How does model handle noisy inputs?
- Stateful models should filter noise

### Adversarial Examples
- Construct inputs that cause state collapse or explosion
- Identify failure modes

---

## 4. Efficiency Metrics

| Metric | Description |
|--------|-------------|
| State Compression Ratio | How much input info retained in state? Compare with theoretical limits |
| Compute per Token | FLOPs per token vs transformer baselines |

---

## 5. Generalization to Unseen Lengths

- Train on sequences of one length
- Test on much longer sequences
- **Expectation:** Stateful models generalize better than attention-window-limited transformers

---

## 6. Multi-Task Learning

- Perform multiple tasks in sequence without forgetting
- Key advantage of stateful models

---

## 7. State Interpretability

- Can we decode the state to understand what model remembers?
- Sanity check for research

---

## Synthetic Benchmark Suite

| Benchmark | Tests |
|-----------|-------|
| ListOps (from Mamba paper) | Long-range reasoning |
| Copying and Selective Copying | Memory |
| Adding Problem | Remember numbers over time |
| Text Generation with Constraints | Follow instructions over long passages |
| Conversational Consistency | Multi-turn dialogue, remember facts |

---

## Comparison Strategy

**Do NOT compare directly with transformer-optimized benchmarks.**

Instead:
1. Compare with baselines (pure RWKV, pure Mamba, transformer) on same tasks
2. Report normalized metrics (perplexity per token, accuracy per task)
3. Report efficiency metrics (memory, compute, training time)
4. Ablation studies showing contribution of each component

---

## Diagnostic Implementation

```python
class DiagnosticWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.state_entropy_history = []
        self.gradient_norms = []

    def forward(self, x):
        output, state = self.model(x)
        entropy = self.compute_entropy(state)
        self.state_entropy_history.append(entropy)
        return output

    def compute_entropy(self, state):
        prob = F.softmax(state, dim=-1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
        return entropy.mean().item()
```

---

## Additional Recommended Tests

### 1. State Coherence Test
- **Input:** Two parallel stories with intersecting characters
- **Test:** Can model keep stories separate in state?

### 2. Incremental Learning Test
- Train on sequence of tasks (A, B, C)
- Test if learning C causes forgetting of A and B

### 3. Real-time Adaptation Test
- Change rules of game/task mid-stream
- Measure how quickly model adapts

### 4. Energy Efficiency
- Measure power consumption or proxy via FLOPs + memory access

### 5. Latency Under Load
- Performance when processing input stream under different load conditions

---

## Priority

Given time/resource constraints, prioritize tests that:
1. Run quickly
2. Give clear signals about strengths/weaknesses
3. Are meaningful for intended use case (conversational model)

---

## Missing Diagnostic Dimensions

*(To be continued...)*

---

## Missing Diagnostic Dimensions (Continued)

### 1. Temporal Gradient Flow Analysis

Most teams measure gradients instantaneously. Stateful models need temporal analysis:

```python
# Track gradient propagation through time
gradient_propagation_time = []
for t in range(sequence_length):
    # Check if gradient from position t affects state at position t+n
    grad_retention = gradient_norm_at_position(t+n, wrt_input(t)) / initial_gradient_norm
    gradient_propagation_time.append(1/grad_retention if grad_retention > 0 else 0)
```

**Insight:** Measures how far back gradients flow - critical for long-term learning.

### 2. State Entropy Spectrum Analysis

Not just entropy, but frequency distribution of state activations:

```python
# Fourier transform of state activations over time
state_fft = torch.fft.rfft(state_sequence, dim=1)
# Low frequency = stable patterns, high frequency = noise/artifacts
signal_to_noise_ratio = state_fft[:, :low_freq_cutoff].norm() / state_fft[:, low_freq_cutoff:].norm()
```

**Detects:** Whether model captures meaningful patterns vs. memorizing noise.

### 3. Architectural Symmetry Breaking

Test if both components are actually contributing:

```python
# Freeze one component, measure performance drop
base_perf = evaluate(model, test_data)
model.freeze_component('RWKV')
perf_without_rwkv = evaluate(model, test_data)
model.unfreeze_all()
model.freeze_component('Mamba')
perf_without_mamba = evaluate(model, test_data)

symmetry_breaking = abs((perf_without_rwkv - base_perf) - (perf_without_mamba - base_perf))
```

- **High symmetry breaking = Good** (components specialized)
- **Low symmetry breaking = Bad** (redundant components)

---

## Benchmark Creation Strategy

Since no benchmarks exist, create canonical tests that become your standard:

### 1. Information Persistence Tests

```python
test_suite = {
    'immediate_recall': (0, 10),      # 0-10 tokens back
    'short_term': (10, 100),          # 10-100 tokens  
    'working_memory': (100, 1000),    # 100-1000 tokens
    'long_term': (1000, 10000),       # 1000-10000 tokens
}
```

Each test measures accuracy decay over distance - plot the curve.

### 2. Importance Discrimination Tests

Create datasets with:
- 90% filler text (unimportant)
- 10% key information (important)

Test if model remembers the 10% much better than the 90%.

### 3. Cross-Domain State Transfer Tests

```python
train_domain = 'scientific_papers'
test_domain = 'conversational_dialogue'
# Measure performance ratio: cross_domain_perf / within_domain_perf
```

Stateful models should transfer better than transformers.

---

## Critical Missing Benchmarks

### A. State Compression Efficiency

```
Input Sequence Length: 10,000 tokens
Model State Size: 1,000 dimensions
Theoretical Max Compression: 10:1
Actual Compression = Information_Rate / State_Size
```

Measure how close you get to theoretical compression limits.

### B. Catastrophic Forgetting Resistance

```python
# Train on task A, then task B, then retest on A
forgetting_ratio = (initial_perf_A - final_perf_A) / initial_perf_A
```

Transformers have ~40-60% forgetting. Your model should be <20%.

### C. Incremental Learning Speed

```
New concept introduced at token 5000
Measure: How many tokens until model incorporates concept?
```

Should be faster than transformers which need full retraining.

---

## Essential Diagnostic Dashboard

Build these real-time monitors:

### 1. State Space Utilization Heatmap

```python
# For each state dimension, track:
# - Activation frequency
# - Information content (mutual information with input)
# - Update frequency
utilization_heatmap = compute_state_utilization(model, batch)
```

Visualize which state dimensions are useful vs. dead.

### 2. Component Contribution Timeline

```python
# Over a sequence, track which component dominates
component_contributions = []
for pos in range(seq_len):
    rwkv_out = model.rwkv_component(hidden_states[:, pos])
    mamba_out = model.mamba_component(hidden_states[:, pos])
    contribution_ratio = rwkv_out.norm() / mamba_out.norm()
    component_contributions.append(contribution_ratio)
```

Shows if components specialize for different parts of sequence.

### 3. Information Bottleneck Analysis

```python
# Measure information flow through architecture
input_information = compute_mutual_info(input, target)
state_information = compute_mutual_info(state, target)
output_information = compute_mutual_info(output, target)

bottleneck_ratio = state_information / input_information
preservation_ratio = output_information / state_information
```

Detects where information is lost.

---

## Novel Evaluation Metrics

Since standard metrics don't apply, define your own:

### 1. State Coherence Score

```python
# Compute similarity between consecutive states
state_similarity = cosine_similarity(state_t, state_{t-1})
# Ideal: Smooth transitions with occasional jumps for new concepts
coherence = 1 - torch.abs(state_similarity.diff()).mean()
```

### 2. Forgetting Appropriateness Score

```python
# Test if model forgets the right things
important_info_recall = recall_of_important_information()
unimportant_info_recall = recall_of_unimportant_information()
appropriateness = important_info_recall - unimportant_info_recall
```

Higher = better at selective forgetting.

### 3. Context Utilization Efficiency

```python
# Compare performance to theoretical maximum
actual_perplexity = compute_perplexity(model, test_data)
theoretical_min_perplexity = estimate_from_data_entropy()
efficiency = theoretical_min_perplexity / actual_perplexity
```

---

## Stress Tests Most Teams Miss

### 1. State Overflow Test

```python
# Feed extremely long sequences until state saturates
sequence_lengths = [100, 1000, 10000, 100000]
for length in sequence_lengths:
    perf = evaluate_on_length(model, length)
    # Plot performance vs length - should degrade gracefully
```

### 2. Mode Collapse Test

```python
# Check if model collapses to using only one component
for batch in test_data:
    outputs = model(batch, return_component_outputs=True)
    if std(outputs['component_ratios']) < threshold:
        print("Mode collapse detected!")
```

### 3. Adversarial State Perturbation

```python
# Test robustness to state corruption
clean_perf = evaluate(model, test_data)
perturbed_state = model.state + noise
model.state = perturbed_state
perturbed_perf = evaluate(model, test_data)
robustness = perturbed_perf / clean_perf
```

---

## Comparative Framework Without Established Benchmarks

### 1. Normalized Performance Index

```
NPI = (Your_Model_Perf - Worst_Possible) / (Theoretical_Max - Worst_Possible)
```

Where "Worst_Possible" is random guessing, "Theoretical_Max" is estimated from data.

### 2. Efficiency-Pareto Frontier

Plot all models (yours, RWKV, Mamba, Transformers) on:
- X-axis: Compute (FLOPs/token)
- Y-axis: Performance (your custom metric)

Your model should dominate the Pareto frontier.

### 3. Capability Radar Chart

6 dimensions:
1. Long-term memory
2. Compute efficiency
3. Training stability
4. Inference speed
5. State utilization
6. Forgetting appropriateness

Plot your model vs baselines.

---

## Implementation Priority

### Week 1-2 (Essential):
- State utilization heatmap
- Component contribution timeline
- Information persistence tests

### Week 3-4 (Advanced):
- State entropy spectrum analysis
- Catastrophic forgetting tests
- Cross-domain transfer tests

### Week 5-6 (Research-grade):
- Temporal gradient flow analysis
- Information bottleneck analysis
- Adversarial state perturbation tests

---

## The "Killer Test": World Model Coherence

The ultimate test for your architecture:

```python
def world_model_coherence_test(model):
    # Feed a coherent world description over 10k tokens
    # Then ask questions that require integrating across the entire description
    # Example: "Given the economic system described at token 1000 
    # and the political system at token 5000, what would happen if X?"
    
    # Score = correctness of integrated reasoning
    return coherence_score
```

**This tests if your model truly builds a "worldview" rather than just remembering facts.**

---

## Documentation Strategy

Since nothing is documented, we must create:

### 1. Diagnostic Dictionary

Each metric must have:
- **Definition** - What it measures
- **Calculation** - Exact formula/code
- **Interpretation** - What values mean

### 2. Failure Mode Registry

Document what each metric detects:
- Observed failure modes
- Which metric caught it
- How to diagnose

### 3. Threshold Values

Establish what's "good" vs "bad" for each metric:

| Metric | Bad | Acceptable | Good |
|--------|-----|------------|------|
| State Coherence | <0.3 | 0.3-0.7 | >0.7 |
| Symmetry Breaking | <0.1 | 0.1-0.3 | >0.3 |
| Forgetting Ratio | >0.4 | 0.2-0.4 | <0.2 |
| Component Ratio Std | <0.05 | 0.05-0.2 | >0.2 |

*(Actual thresholds TBD through experimentation)*

### 4. Cross-Metric Correlations

Document how metrics relate:
- If X is high and Y is low, what does that indicate?
- Which metrics move together?
- Which are independent signals?

---

## Final Critical Addition: Compute-Performance Curves

Most teams measure final performance only. **You must measure:**

```
Performance = f(Compute Budget)
```

For multiple compute budgets (1M, 10M, 100M, 1B FLOPs), which architecture gives best performance?

| Compute Budget | Arch 1 Perf | Arch 2 Perf | Arch 3 Perf |
|----------------|-------------|-------------|-------------|
| 1M FLOPs | ? | ? | ? |
| 10M FLOPs | ? | ? | ? |
| 100M FLOPs | ? | ? | ? |
| 1B FLOPs | ? | ? | ? |

**This is your ultimate competitive advantage** - showing your architecture reaches acceptable performance with less compute.

---

## III. Performance Benchmarking Suite

**Purpose:** Benchmark RWKV-6 and Mamba-2 kernel performance to validate CUDA optimizations and measure throughput improvements.

**Cross-Reference:** See [V4.5_CUDA_KERNELS.md](V4.5_CUDA_KERNELS.md) for kernel implementation details and optimization strategies.

### Kernel Benchmark Class

Complete benchmarking suite for CUDA kernel validation:

```python
import time
import numpy as np
from torch.cuda.amp import autocast

class KernelBenchmark:
    """Benchmark RWKV-6 and Mamba-2 kernels"""
    
    @staticmethod
    def benchmark_rwkv6(model, batch_size=8, seq_len=2048, warmup=10, trials=50):
        """Benchmark RWKV-6 forward pass"""
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, model.hidden_size, device=device)
        state = None
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                output, new_state = model(x, state)
                torch.cuda.synchronize()
        
        # Benchmark
        times = []
        torch.cuda.synchronize()
        
        for _ in range(trials):
            start = time.perf_counter()
            
            with torch.no_grad(), autocast(enabled=True):
                output, new_state = model(x, state)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        # Calculate statistics
        times = np.array(times)
        avg_time = times.mean()
        std_time = times.std()
        tokens_per_sec = (batch_size * seq_len) / (avg_time / 1000)
        
        return {
            'avg_ms': avg_time,
            'std_ms': std_time,
            'tokens_per_sec': tokens_per_sec,
            'tokens_per_sec_per_param': tokens_per_sec / sum(p.numel() for p in model.parameters())
        }
    
    @staticmethod
    def profile_memory(model, batch_size=8, seq_len=2048):
        """Profile GPU memory usage"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        device = next(model.parameters()).device
        x = torch.randn(batch_size, seq_len, model.d_model, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        torch.cuda.synchronize()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return {
            'memory_gb': memory_used,
            'memory_per_token_mb': (memory_used * 1024) / (batch_size * seq_len)
        }
```

### Usage Example

```python
# Benchmark PyTorch prototype vs CUDA kernels
from rwkv6_prototype import RWKV6Attention_Prototype
from hybrid_v4 import RWKV6Attention_CUDA

# Create models
prototype = RWKV6Attention_Prototype(hidden_size=512).cuda()
cuda_model = RWKV6Attention_CUDA(hidden_size=512).cuda()

# Benchmark
proto_results = KernelBenchmark.benchmark_rwkv6(prototype, seq_len=2048)
cuda_results = KernelBenchmark.benchmark_rwkv6(cuda_model, seq_len=2048)

print(f"Prototype: {proto_results['tokens_per_sec']:.0f} tok/s")
print(f"CUDA: {cuda_results['tokens_per_sec']:.0f} tok/s")
print(f"Speedup: {cuda_results['tokens_per_sec'] / proto_results['tokens_per_sec']:.1f}x")

# Memory profiling
proto_mem = KernelBenchmark.profile_memory(prototype, seq_len=2048)
cuda_mem = KernelBenchmark.profile_memory(cuda_model, seq_len=2048)

print(f"Prototype memory: {proto_mem['memory_gb']:.2f} GB")
print(f"CUDA memory: {cuda_mem['memory_gb']:.2f} GB")
```

### Expected Performance Targets

**From V4.5_CUDA_KERNELS.md:**

| Implementation | Seq Len | Tokens/Sec | Speedup vs Prototype |
|----------------|---------|------------|----------------------|
| PyTorch Prototype | 256 | ~5-10K | 1.0x (baseline) |
| PyTorch Prototype | 1024 | ~2-5K | - |
| Mamba-2 Official CUDA | 256 | ~100-200K | **10-20x** |
| Mamba-2 Official CUDA | 1024 | ~80-150K | **15-30x** |
| RWKV-6 Custom CUDA | 256 | TBD | **Target: 30-50x** |
| RWKV-6 Custom CUDA | 1024 | TBD | **Target: 30-50x** |
| Full CUDA Hybrid | 256 | **100-300K** | **Target: 20-30x** |

### Integration with Monitoring Tools

**Link to Task 6.5:** Test monitoring during kernel benchmarks

```bash
# Terminal 1: Start monitoring
nvidia-smi damon -i 0 -f benchmark_gpu_monitor.csv &
MONITOR_PID=$!

# Terminal 2: Run benchmark
python -c "
from kernel_benchmark import KernelBenchmark
from hybrid_v4 import HybridLanguageModel
model = HybridLanguageModel().cuda()
results = KernelBenchmark.benchmark_rwkv6(model, seq_len=1024, trials=100)
print(results)
"

# Stop monitoring
kill $MONITOR_PID
```

### Profiling with NSight Systems

**Advanced profiling for kernel optimization:**

```bash
# Profile CUDA kernel execution
nsys profile -o rwkv6_profile \
    --stats=true \
    --force-overwrite=true \
    python benchmark_kernels.py

# View results
nsys stats rwkv6_profile.qdrep

# Key metrics to check:
# - Kernel launch overhead
# - Memory bandwidth utilization
# - SM occupancy
# - Warp stall reasons
```

### Benchmark Validation Gates

**Before declaring kernel optimization complete:**

1. ✅ **Throughput:** >10x speedup over PyTorch prototype
2. ✅ **Memory:** No memory leaks, <1.5x prototype VRAM usage
3. ✅ **Stability:** Zero NaN/Inf in 1000-trial benchmark
4. ✅ **Consistency:** Std dev <5% of mean latency
5. ✅ **GPU Utilization:** >80% during compute phases
6. ✅ **Numerical Accuracy:** Output matches prototype within 1e-3

**See V4.5_CUDA_KERNELS.md for full validation protocol.**

---

*End of V4 Diagnostics Document*
