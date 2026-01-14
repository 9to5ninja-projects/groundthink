# GroundThink Implementation Guide: PyTorch Path

**Date:** January 13, 2026  
**Target Hardware:** RTX 4050 Ada Lovelace (6GB VRAM, 20 SMs)  
**Framework:** PyTorch (torch.compile for optimization)  
**Prerequisites:** groundthink_arbiter_research.md, groundthink_parallel_scan_math.md

---

## Hardware Profile: RTX 4050 Ada Lovelace

| Spec | Value | Implication |
|------|-------|-------------|
| VRAM | 6GB | Aggressive chunking for long contexts |
| CUDA Cores | 2560 | Good parallel throughput |
| Tensor Cores | 4th Gen | Native BF16/FP8 acceleration |
| Shared Memory | 48KB/SM (100KB max) | Comfortable for scan intermediates |
| L2 Cache | 32MB | State caching friendly |
| Memory Bandwidth | 192 GB/s | Primary bottleneck |
| SMs | 20 | Block size tuning target |

**Primary constraint:** Memory bandwidth, not compute. Design kernels to maximize data reuse.

---

## PyTorch Implementation: Parallel Fusion Arbiter

### Basic Structure

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelFusionArbiter(nn.Module):
    """
    GLU-style arbiter for Mamba-2 / RWKV-6 fusion.
    Removes sequential h_{t-1} dependency for parallel execution.
    """
    def __init__(self, d_model: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        
        # Pre-arbiter normalization (critical for State Duality Sync)
        self.norm_mamba = nn.RMSNorm(d_model)
        self.norm_rwkv = nn.RMSNorm(d_model)
        
        # Gating projections (input-conditioned, not state-conditioned)
        self.gate_mamba = nn.Linear(d_model, d_model, bias=bias)
        self.gate_rwkv = nn.Linear(d_model, d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
    def forward(
        self, 
        h_mamba: torch.Tensor,  # (batch, seq, d_model)
        h_rwkv: torch.Tensor,   # (batch, seq, d_model)
        x: torch.Tensor         # (batch, seq, d_model) - original input
    ) -> torch.Tensor:
        # State Duality Sync: normalize before fusion
        h_m = self.norm_mamba(h_mamba)
        h_r = self.norm_rwkv(h_rwkv)
        
        # Input-conditioned gates (parallel across sequence)
        g_m = torch.sigmoid(self.gate_mamba(x))
        g_r = torch.sigmoid(self.gate_rwkv(x))
        
        # Parallel fusion
        fused = g_m * h_m + g_r * h_r
        
        return self.out_proj(fused)
```

### With Linearized Recurrence (minGRU-style)

For cases where you want temporal gating but still need parallelism:

```python
class ParallelScanArbiter(nn.Module):
    """
    minGRU-style arbiter with parallel scan capability.
    Implements: h_t = a_t * h_{t-1} + b_t * x_t
    where the recurrence is computed via parallel prefix scan.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Normalization
        self.norm_mamba = nn.RMSNorm(d_model)
        self.norm_rwkv = nn.RMSNorm(d_model)
        
        # Gate projections for linearized recurrence
        # a_t: forget gate (how much to retain)
        # b_t: input gate (how much new info to add)
        self.proj_a = nn.Linear(d_model, d_model)
        self.proj_b = nn.Linear(d_model, d_model)
        
        # Branch weighting
        self.branch_gate = nn.Linear(d_model, 2)  # outputs [w_mamba, w_rwkv]
        
        self.out_proj = nn.Linear(d_model, d_model)
    
    def parallel_scan(
        self, 
        a: torch.Tensor,  # (batch, seq, d_model) - forget gates
        b: torch.Tensor,  # (batch, seq, d_model) - input gates
        x: torch.Tensor   # (batch, seq, d_model) - inputs
    ) -> torch.Tensor:
        """
        Compute h_t = a_t * h_{t-1} + b_t * x_t via parallel scan.
        
        Uses associative property:
        (a_2, b_2*x_2) ⊗ (a_1, b_1*x_1) = (a_2*a_1, a_2*b_1*x_1 + b_2*x_2)
        """
        batch, seq_len, d = x.shape
        
        # Clamp gates for numerical stability
        a = torch.sigmoid(a)  # (0, 1) range
        b = torch.sigmoid(b)
        
        # Compute b * x once
        bx = b * x
        
        # For now: use cumulative product formulation
        # This is O(N) but torch.compile can optimize
        # True parallel scan requires custom kernel
        
        # Cumulative product of forget gates
        log_a = torch.log(a + 1e-8)
        cumsum_log_a = torch.cumsum(log_a, dim=1)
        cum_a = torch.exp(cumsum_log_a)  # (batch, seq, d)
        
        # Scale inputs by cumulative forget
        # h_t = sum_{i=1}^{t} (prod_{j=i+1}^{t} a_j) * b_i * x_i
        scaled_bx = bx / (cum_a + 1e-8)
        cumsum_scaled = torch.cumsum(scaled_bx, dim=1)
        h = cum_a * cumsum_scaled
        
        return h
    
    def forward(
        self,
        h_mamba: torch.Tensor,
        h_rwkv: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        # Normalize branches
        h_m = self.norm_mamba(h_mamba)
        h_r = self.norm_rwkv(h_rwkv)
        
        # Compute branch weights
        weights = F.softmax(self.branch_gate(x), dim=-1)
        w_m, w_r = weights[..., 0:1], weights[..., 1:2]
        
        # Weighted combination as input to scan
        combined = w_m * h_m + w_r * h_r
        
        # Compute temporal gates
        a = self.proj_a(x)
        b = self.proj_b(x)
        
        # Parallel scan over fused representation
        h_out = self.parallel_scan(a, b, combined)
        
        return self.out_proj(h_out)
```

---

## torch.compile Optimization

### Basic Compilation

```python
# Compile with reduce-overhead for inference
model = torch.compile(model, mode="reduce-overhead")

# For training, use default mode
model = torch.compile(model)

# For maximum optimization (slower compile, faster runtime)
model = torch.compile(model, mode="max-autotune")
```

### Compilation Modes Comparison

| Mode | Compile Time | Runtime | Use Case |
|------|--------------|---------|----------|
| `default` | Medium | Good | Training |
| `reduce-overhead` | Fast | Better | Inference |
| `max-autotune` | Slow | Best | Production deployment |

### Enabling Tensor Cores

```python
# Enable TF32 for matmuls (Ada Lovelace native)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Or use BF16 for better throughput
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(h_mamba, h_rwkv, x)
```

---

## Memory Management for 6GB VRAM

### Chunked Processing

```python
def chunked_forward(
    model: nn.Module,
    h_mamba: torch.Tensor,
    h_rwkv: torch.Tensor,
    x: torch.Tensor,
    chunk_size: int = 2048
) -> torch.Tensor:
    """Process long sequences in chunks to fit in VRAM."""
    batch, seq_len, d = x.shape
    outputs = []
    
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        chunk_out = model(
            h_mamba[:, i:end],
            h_rwkv[:, i:end],
            x[:, i:end]
        )
        outputs.append(chunk_out)
    
    return torch.cat(outputs, dim=1)
```

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientArbiter(ParallelFusionArbiter):
    def forward(self, h_mamba, h_rwkv, x):
        # Checkpoint the fusion computation
        return checkpoint(
            super().forward,
            h_mamba, h_rwkv, x,
            use_reentrant=False
        )
```

### Estimated Memory Budget (6GB)

| Component | Size (d=1024, seq=4096, batch=1) |
|-----------|----------------------------------|
| h_mamba | 16 MB |
| h_rwkv | 16 MB |
| x | 16 MB |
| Gates (g_m, g_r) | 32 MB |
| Intermediates | ~100 MB |
| Model weights | ~50 MB |
| **Total forward** | ~230 MB |
| **With gradients** | ~700 MB |
| **Headroom** | ~5.3 GB |

You have comfortable headroom. Can likely push to seq=16384 or batch=4 before hitting limits.

---

## Validation Tests

### Test 1: Gradient Flow Comparison

```python
def test_gradient_flow():
    """Verify gradients reach both branches effectively."""
    d_model = 256
    seq_len = 1024
    batch = 2
    
    model = ParallelFusionArbiter(d_model).cuda()
    
    # Random inputs
    h_mamba = torch.randn(batch, seq_len, d_model, requires_grad=True, device='cuda')
    h_rwkv = torch.randn(batch, seq_len, d_model, requires_grad=True, device='cuda')
    x = torch.randn(batch, seq_len, d_model, device='cuda')
    
    # Forward + backward
    out = model(h_mamba, h_rwkv, x)
    loss = out.sum()
    loss.backward()
    
    # Check gradient magnitudes
    grad_mamba = h_mamba.grad.abs().mean().item()
    grad_rwkv = h_rwkv.grad.abs().mean().item()
    ratio = grad_mamba / grad_rwkv
    
    print(f"Mamba gradient magnitude: {grad_mamba:.6f}")
    print(f"RWKV gradient magnitude:  {grad_rwkv:.6f}")
    print(f"Ratio (Mamba/RWKV):       {ratio:.3f}")
    
    # Healthy ratio should be close to 1.0
    # If Mamba Paradox persists, ratio will be >> 1
    assert 0.1 < ratio < 10, f"Gradient imbalance detected: {ratio}"
    print("✓ Gradient flow test passed")
```

### Test 2: Branch Contribution Analysis

```python
def test_branch_contributions():
    """Measure how much each branch contributes to output."""
    d_model = 256
    seq_len = 1024
    batch = 2
    
    model = ParallelFusionArbiter(d_model).cuda()
    model.eval()
    
    h_mamba = torch.randn(batch, seq_len, d_model, device='cuda')
    h_rwkv = torch.randn(batch, seq_len, d_model, device='cuda')
    x = torch.randn(batch, seq_len, d_model, device='cuda')
    
    with torch.no_grad():
        # Full output
        out_full = model(h_mamba, h_rwkv, x)
        
        # Zero out each branch
        out_no_mamba = model(torch.zeros_like(h_mamba), h_rwkv, x)
        out_no_rwkv = model(h_mamba, torch.zeros_like(h_rwkv), x)
        
        # Contribution = change when branch removed
        mamba_contrib = (out_full - out_no_mamba).abs().mean().item()
        rwkv_contrib = (out_full - out_no_rwkv).abs().mean().item()
        
        total = mamba_contrib + rwkv_contrib
        mamba_pct = 100 * mamba_contrib / total
        rwkv_pct = 100 * rwkv_contrib / total
    
    print(f"Mamba contribution: {mamba_pct:.1f}%")
    print(f"RWKV contribution:  {rwkv_pct:.1f}%")
    
    # Flag if one branch dominates excessively
    if mamba_pct < 5 or mamba_pct > 95:
        print("⚠ Warning: Severe branch imbalance")
    else:
        print("✓ Branch contribution test passed")
```

### Test 3: Throughput Benchmark

```python
import time

def benchmark_throughput():
    """Measure tokens/second on target hardware."""
    d_model = 1024
    seq_len = 4096
    batch = 1
    n_warmup = 10
    n_runs = 50
    
    model = ParallelFusionArbiter(d_model).cuda()
    model = torch.compile(model, mode="reduce-overhead")
    model.eval()
    
    h_mamba = torch.randn(batch, seq_len, d_model, device='cuda')
    h_rwkv = torch.randn(batch, seq_len, d_model, device='cuda')
    x = torch.randn(batch, seq_len, d_model, device='cuda')
    
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(h_mamba, h_rwkv, x)
    torch.cuda.synchronize()
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(h_mamba, h_rwkv, x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    tokens_per_sec = (batch * seq_len * n_runs) / elapsed
    ms_per_batch = (elapsed / n_runs) * 1000
    
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"Latency:    {ms_per_batch:.2f} ms/batch")
    print(f"Seq length: {seq_len}, Batch: {batch}, d_model: {d_model}")
```

---

## Integration Checklist

- [ ] Implement `ParallelFusionArbiter` base class
- [ ] Add RMSNorm pre-fusion (State Duality Sync)
- [ ] Run gradient flow test—verify ratio near 1.0
- [ ] Run branch contribution test—verify neither branch < 5%
- [ ] Benchmark throughput on RTX 4050
- [ ] Compare against original GRU arbiter metrics
- [ ] If parallel scan needed, implement `ParallelScanArbiter`
- [ ] Profile memory usage, adjust chunk_size if needed
- [ ] Test with torch.compile modes

---

## Future Optimization: Triton Kernel

When PyTorch path is validated and you need maximum inference speed:

```python
# Placeholder for future Triton implementation
import triton
import triton.language as tl

@triton.jit
def parallel_scan_kernel(
    # ... kernel implementation
):
    """True O(log N) parallel scan on GPU."""
    pass
```

Triton kernel development deferred until:
1. Architecture validated via PyTorch path
2. Mamba Paradox confirmed resolved
3. Inference latency becomes bottleneck

---

*Document generated for GroundThink development archive*  
*Implementation target: PyTorch 2.x with torch.compile*
