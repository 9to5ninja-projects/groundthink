# GroundThink Arbiter Architecture Research Notes

**Date:** January 13, 2026  
**Project:** GroundThink (RWKV-6 + Mamba-2 Hybrid Fusion)  
**Phase:** Post-Phase 0 (Base Model Characterization) → Phase 1 Implementation  
**Focus:** Arbiter Design Optimization

---

## Executive Summary

Moving from a standard GRU to a Gated Linear Unit (GLU) or minGRU for the arbiter represents a significant architectural upgrade for the Mamba-2/RWKV-6 fusion. The standard GRU's sequential nature and hidden-state bottleneck creates latency liability in a parallel fusion architecture.

**Core Insight:** In 2026 hardware-lite design, the Arbiter shouldn't just be a switch—it should be a *feature mixer* that maintains the linear complexity of both main branches.

---

## Why GLU/minGRU Beats Standard GRU

### 1. Parallelism
- Standard GRU is inherently sequential: **O(N)**
- This slows overall throughput of parallel branches
- GLU and minGRU can be computed in parallel across the sequence
- Matches the native speed of Mamba-2

### 2. Memory Efficiency
- minGRU removes hidden-state-to-hidden-state dependencies (h_{t-1} dependency)
- Enables **Parallel Scan** optimization
- Prevents the Arbiter from becoming the bottleneck for hardware-lite deployment

### 3. The Bottleneck Question
If current GRU implementation causes lag, it's likely due to the **h_{t-1} dependency**. Switching to gate-only fusion or minGRU removes this dependency and lets hardware Tensor Cores do the heavy lifting.

---

## Reference Implementation: Parallel Fusion Arbiter (GLU-style)

Lightweight implementation for fusing hidden states of Mamba-2 and RWKV-6. Uses learned gating mechanism to decide blend ratio: "logic" (Mamba) vs. "nuance" (RWKV).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelFusionArbiter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Gate to decide the blend ratio based on input features
        self.gate_mamba = nn.Linear(d_model, d_model)
        self.gate_rwkv = nn.Linear(d_model, d_model)
        
        # Optional: Final projection to mix the fused features
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, h_mamba, h_rwkv, x_input):
        """
        h_mamba: Hidden state from Mamba-2 branch
        h_rwkv:  Hidden state from RWKV-6 branch
        x_input: The original input embedding (provides context for gating)
        """
        # Compute weights based on current context/input
        # Sigmoid ensures values stay between 0 and 1
        g_m = torch.sigmoid(self.gate_mamba(x_input))
        g_r = torch.sigmoid(self.gate_rwkv(x_input))
        
        # Parallel Fusion: Mix the 'brains'
        # Allows model to selectively use Mamba's logic or RWKV's flow
        fused_state = (g_m * h_mamba) + (g_r * h_rwkv)
        
        return self.output_proj(fused_state)
```

### Key Design Notes
- **Input-conditioned gating:** Gates computed from `x_input`, not recurrent state
- **Parallel-friendly:** No sequential dependencies between timesteps
- **Dual sigmoid gates:** Independent control over each branch contribution
- **Output projection:** Final mixing layer for fused representation

---

## Hardware Optimization Tips (2026 Target)

### 1. Parallel Scan for the Arbiter
If using a recurrent Arbiter (like minGRU), implement **Prefix Sum / Parallel Scan** algorithm:
- Gating decisions for 1,000-token sequence: **O(log N)** on GPU/NPU
- Standard recurrent: **O(N)**
- Significant speedup for long-context inference

### 2. State Duality Sync
**Critical:** Ensure Mamba-2 and RWKV-6 branches share the same normalization layer (e.g., RMSNorm) **before** entering the Arbiter.

**Why:** Prevents one "brain" from numerically overpowering the other due to different scaling. This directly addresses the "Mamba Paradox" where Mamba receives 10x larger gradients but contributes less than 0.3% to model output.

### 3. Cross-Distillation Training
If training from scratch, try **Cross-Distillation**:
- Train Mamba-2 branch to predict RWKV-6 hidden states
- Train RWKV-6 branch to predict Mamba-2 hidden states
- Creates **"latent synergy"**—branches begin to speak the same internal language
- Makes the Arbiter's job significantly easier at inference

---

## Architectural Comparison

| Aspect | Standard GRU | GLU/minGRU |
|--------|--------------|------------|
| Time Complexity | O(N) sequential | O(N) parallel or O(log N) with scan |
| Hidden State Dependency | h_{t-1} required | Removed |
| GPU Utilization | Poor (sequential ops) | Good (parallel ops) |
| Memory Footprint | Higher (state caching) | Lower (stateless gates) |
| Bottleneck Risk | High | Low |
| Tensor Core Compatibility | Limited | Full |

---

## Connection to Existing GroundThink Findings

### The Mamba Paradox (from Phase 0)
- Mamba receives 10x larger gradients
- Yet contributes less than 0.3% to model output
- **Hypothesis:** Arbiter bottleneck may be contributing to gradient imbalance

### Twin Debate Architecture
- GRU Arbiter was original design for mediating branch "debate"
- GLU-style Arbiter maintains debate semantics while removing sequential bottleneck
- Residual connections remain compatible

### Hardware-Lite Goals
- Target: Consumer-grade inference (RTX 3060/4060 class)
- Sequential GRU operations waste CUDA cores
- Parallel GLU maximizes hardware utilization

---

## Future Test Cases

### 1. Latency Profiling
- Benchmark GRU vs GLU Arbiter on varying sequence lengths (512, 1024, 2048, 4096)
- Measure tokens/second throughput
- Profile memory usage per batch

### 2. Gradient Flow Analysis
- Compare gradient magnitudes through GRU vs GLU paths
- Test if GLU helps resolve Mamba Paradox gradient imbalance
- Monitor branch contribution ratios with each architecture

### 3. Quality Metrics
- Perplexity comparison on held-out test set
- Long-context coherence evaluation
- Branch specialization emergence (does one branch dominate?)

### 4. Ablation Studies
- GLU with vs without output projection
- Shared vs independent gate weights
- Effect of pre-Arbiter normalization (RMSNorm vs LayerNorm)

---

## Open Questions

1. **Loss Performance Comparison:** How does mathematical loss differ between GRU-arbiter and Parallel-Scan-Arbiter for long-context tasks?

2. **Gradient Routing:** Does GLU improve gradient flow to the under-contributing Mamba branch?

3. **Emergent Specialization:** Do branches develop cleaner specialization (logic vs nuance) with parallel gating?

4. **Inference Batching:** What's the optimal batch size for GLU vs GRU on target hardware?

---

## References & Related Work

- minGRU: Minimalist GRU variants with parallel scan compatibility
- Mamba-2: State space models with selective scan
- RWKV-6: Linear attention with channel mixing
- GLU variants: SwiGLU, GeGLU, ReGLU in modern transformer architectures

---

*Document generated for GroundThink development archive*
*Next steps: Implement ParallelFusionArbiter and run comparative benchmarks*
