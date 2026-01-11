# GroundedMamba Foundation

## Core Architecture

**Two components working together:**
1. **StableStateSSM** - Dynamic "thinking" with input-dependent selection
2. **RWKVGroundedMemory** - Stable "grounding" with explicit memory bank

**Key innovation:** Not a theoretical hybrid. Built from what actually works in practice.

---

## Critical Implementation Details

### 1. Discretization (WHERE MOST FAIL)
```
Use first-order Taylor expansion: A_d = I + A * dt
NOT exact matrix exponential (numerically unstable)
```

### 2. State Normalization (NON-NEGOTIABLE)
```
Normalize states after EVERY update
Use LayerNorm, not just clipping
```

### 3. Stability Mechanisms
- Spectral normalization to force eigenvalues < 1
- dt clamped to [0.001, 0.1]
- wkv_state.detach() to prevent gradient explosion
- State norm penalty in loss function

### 4. Memory Updates (SLOW)
```
alpha = 0.01  # 1% blend per update
Update frequency: every 100 steps, not every step
```

### 5. Separate Optimizers (DIFFERENT LRs)
```
SSM params:   lr * 0.5, weight_decay=0.1
RWKV params:  lr * 0.3, weight_decay=0.01
Other params: lr * 1.0, weight_decay=0.1
```

### 6. Gradient Clipping (NON-NEGOTIABLE)
```
max_norm = 1.0
Always. No exceptions.
```

---

## Training Protocol

### Start Small
```python
model = GroundedMamba(
    vocab_size=10000,
    dim=512,      # Small
    depth=8,      # Shallow
    ssm_dim=16,   # Tiny state
    rwkv_heads=4  # Few heads
)
```

### Monitor These Metrics
| Metric | Healthy Range | Action if Violated |
|--------|---------------|-------------------|
| Loss | Smooth decrease | Check LR, data |
| State norms | 0.1 - 10.0 | Increase norm penalty |
| Gradient norms | < 1.0 | Reduce LR |
| Memory usage | O(1) with seq len | Check for leaks |

### Validation Requirements
- Test on actual long sequences (>10k tokens)
- Not just theoretical metrics
- Real-world coherence checks

---

## Known Failure Modes

### RWKV Drift (>10k tokens)
- **Cause:** State accumulates noise over time
- **Solution:** Explicit memory bank with slow updates, state normalization

### Mamba Over-Selectivity
- **Cause:** Forgets important context too aggressively
- **Solution:** Grounding from RWKV memory, dt clamping

### Training Instability
- **Cause:** SSM sensitive to initialization/LR
- **Solution:** Separate optimizers, proper init, grad clipping

### State Explosion
- **Cause:** Unbounded state growth
- **Solution:** LayerNorm after every update, spectral normalization

---

## Architecture Constants

### GroundedMambaBlock
- Fixed gate ratio during early training: SSM 0.7, RWKV 0.3
- Learned gate exists but disabled initially
- Gate history buffer for debugging

### RWKVGroundedMemory
- Geometric decay across heads: decay = exp(-5.0 * (h+1) / num_heads)
- Memory size: 1024 (explicit key-value store)
- Time decay and time_first as learned parameters

### StableStateSSM
- A matrix initialized: -I * 0.1 + noise * 0.01 (ensures negative eigenvalues)
- dt_proj bias: [1.0, 0.5] (initial "thinking speed")
- B, C projections: small weights (std=0.01) to prevent explosion

---

## Debugging Procedures (CRITICAL)

### 1. Gradient Explosion Detection
```python
# Add to training loop
if torch.isnan(loss) or torch.isinf(loss):
    logger.error(f"NaN/Inf loss at step {self.step}")
    
    for name, param in self.model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                logger.error(f"NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                logger.error(f"Inf gradient in {name}")
    
    # Reduce learning rate and continue
    for param_group in self.optimizer.param_groups:
        param_group['lr'] *= 0.5
```

### 2. State Drift Detection
```python
def check_state_drift(model, threshold=10.0):
    """Process 4096 tokens in 512-token chunks, compare state drift"""
    with torch.no_grad():
        test_seq = torch.randint(0, 10000, (1, 4096))
        states_history = []
        
        for i in range(0, 4096, 512):
            chunk = test_seq[:, i:i+512]
            _, states = model(chunk)
            states_history.append(states)
        
        for i in range(1, len(states_history)):
            for name in states_history[0]:
                state1 = states_history[i-1][name]
                state2 = states_history[i][name]
                if state1 is not None and state2 is not None:
                    drift = torch.norm(state2 - state1) / torch.norm(state1)
                    if drift > threshold:
                        logger.warning(f"Large drift in {name}: {drift:.2f}")
                        return True
        return False
```

### 3. Memory Leak Detection
```python
def check_memory_leak(model, iterations=100):
    """Run 100 small forward passes, check for 50%+ memory growth"""
    import gc
    initial_memory = torch.cuda.memory_allocated()
    
    for i in range(iterations):
        x = torch.randint(0, 10000, (1, 128)).cuda()
        model(x)
        gc.collect()
        torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated()
    if final_memory > initial_memory * 1.5:
        logger.error(f"Memory leak: {initial_memory/1024**2:.1f}MB -> {final_memory/1024**2:.1f}MB")
        return True
    return False
```

---

## Training Loop Components

### GradientDebugger Class
- Register hooks on all parameters
- Track: mean, std, max, min of gradients
- Count NaN/Inf occurrences
- Replace NaN/Inf gradients with zeros
- Log stats every 100 steps

### StateMonitor Class
- Record state norms, mean, std, max, min
- Warn if state norm > 100
- Error on NaN/Inf in states
- Track gate history if available

### Loss Function Components
```python
# 1. Base cross-entropy
ce_loss = F.cross_entropy(logits, targets, ignore_index=-100)

# 2. State stability penalty (training only)
for name, state in states.items():
    state_norm = torch.norm(state, dim=-1).mean()
    target_norm = 5.0
    stability_loss += F.mse_loss(state_norm, target_norm)

# 3. Output consistency penalty
output_diff = logits[:, 1:] - logits[:, :-1]
consistency_loss = output_diff.norm(dim=-1).mean() * 0.01

total_loss = ce_loss + 0.1 * stability_loss + consistency_loss
```

### Scheduler: Warmup + Cosine Decay
```python
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
```

### Long Context Stability Test
```python
# Test at: 1024, 4096, 8192, 16384 tokens
# Measure: perplexity, inference time, avg state norm, tokens/sec
# Reset states between tests
```

---

## Quick Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss spikes | LR too high | Reduce LR by 2× |
| NaN loss | Gradient explosion | Check grads, reduce LR |
| Memory grows | retain_graph=True | Clear cache, check code |
| State explosion | Unbounded growth | Increase norm penalty, reduce dt_max |
| Slow training | Small batch | Increase batch, use grad accumulation |
| Val diverges | Overfitting | More regularization, smaller model |

---

## Environment Setup
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTHONFAULTHANDLER=1
```

---

## Advanced Recovery & Debugging Tools

### EmergencyRecovery (02_emergency_recovery.py)
- Keeps rolling buffer of last 5 good states
- Auto-rollback when loss explodes
- Reduces LR by 0.7× on recovery

### Context Shift Detection (03_context_shift_detection.py)
- Monitors cosine similarity between recent hidden states
- Resets states when similarity < 0.8 (topic change detected)
- Prevents carrying over irrelevant context

### GradientPathologyDetector (04_gradient_pathology_detector.py)
- Detects: gradient sparsity (>90% zeros = dying weights)
- Detects: direction flip-flop (cosine sim < -0.5)
- Detects: magnitude collapse (<1e-7)
- Triggers alert at pathology score > 5

### Unfreeze Brain (05_unfreeze_brain.py)
Three methods to escape bad local minima:
- `soft_reset`: Add small noise to state params
- `selective_reinit`: Reinitialize stuck SSM A matrices
- `gradient_injection`: Inject synthetic gradients to dead params

### MemoryEfficientCheckpointer (06_memory_efficient_checkpointer.py)
- Sharded saving (2GB per shard) to avoid OOM
- Separate optimizer state storage
- Metadata tracking (params, dtype, timestamp)

### Meaningful Validation (07_meaningful_validation.py)
Beyond perplexity:
- `state_coherence`: Do states evolve smoothly?
- `context_retention`: Can it use info from earlier?
- `reasoning_accuracy`: Simple logic tests
- `drift_score`: How much do states drift?

### Production Checklist (08_production_checklist.py)
Run before deploying:
- `state_reset_works`: Does reset_states() actually reset?
- `memory_scales_O1`: Memory sublinear with seq length?
- `inference_stable`: Variance < 0.01 over 100 runs?
- `no_memory_leaks`: No memory growth over iterations?
- `handles_edge_cases`: Empty input, single token, etc.
- `reproducible`: Same output with same seed?

---

## Configuration Reference (09_config_actual.yaml)

```yaml
model:
  dim: 2048
  depth: 24
  ssm_state_dim: 64
  rwkv_heads: 16
  rwkv_head_dim: 128
  
  stability:
    dt_min: 0.001
    dt_max: 0.1
    state_norm_clip: 10.0
    gradient_clip: 1.0
    state_reset_threshold: 0.3
    
  memory:
    size: 4096
    update_rate: 0.01
    retrieval_heads: 4
    
  training:
    batch_size: 32
    gradient_accumulation: 4
    learning_rate: 3e-4
    warmup_steps: 2000
    total_steps: 100000
    
    loss_weights:
      ce: 1.0
      state_stability: 0.1
      memory_consistency: 0.05
```

---

## File Structure
```
E:\RWKV\groundthink\
├── FOUNDATION.md          # This file - DO NOT DEVIATE
├── model.py               # Core model implementation
├── layers.py              # Layer implementations
├── training.py            # Training utilities (RealTrainingLoop)
├── config.py              # Configuration
├── test_model.py          # Validation tests
└── configs/
    └── minimal.yaml       # Starting config (512 dim, 8 depth)
```

---

## Reference Files (deepseek_design/more/)
| File | Purpose |
|------|---------|
| 01_real_training_loop.py | Complete training loop with debugging |
| 02_emergency_recovery.py | Auto-rollback on loss explosion |
| 03_context_shift_detection.py | Topic change detection |
| 04_gradient_pathology_detector.py | Dying weights/flip-flop detection |
| 05_unfreeze_brain.py | Escape bad local minima |
| 06_memory_efficient_checkpointer.py | Sharded checkpoint saving |
| 07_meaningful_validation.py | Beyond-perplexity metrics |
| 08_production_checklist.py | Pre-deployment validation |
| 09_config_actual.yaml | Production config template |

---

## FINAL ADVICE (from DeepSeek)

1. **Start with smallest possible model (~10M params)**. Get it perfect before scaling.

2. **Log EVERYTHING.** When it breaks (and it will), you need the logs.

3. **Build in automatic recovery.** Don't rely on manual intervention.

4. **Test on ACTUAL long sequences (>10k tokens) from day one.**

5. **Keep a "golden" checkpoint that you know works. NEVER overwrite it.**

> "These details are what separate a working prototype from a production system."

---

## Next Steps (IN ORDER)
1. Save the complete model implementation
2. Create minimal test script
3. Validate on small dataset
4. Get loss decreasing smoothly
5. THEN consider scaling

**DO NOT SKIP STEPS. DO NOT ADD COMPLEXITY BEFORE VALIDATION.**
