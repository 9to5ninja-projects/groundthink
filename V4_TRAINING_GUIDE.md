# V4 Training Guide
> **GOLD STANDARD** - This is THE authoritative reference for how we train hybrid models.  
> Distilled from V3_CROSS_REFERENCE.md and prior experiments. Read this FIRST.

**Status:** Active | **Version:** 1.1 | **Last Updated:** 2026-01-09

---

## � EDITING GUIDELINES FOR AGENTS

**Make small, incremental edits. DO NOT create massive documents in one operation.**

Large edits (500+ lines) frequently timeout, get truncated, or introduce errors. Break work into 50-150 line sections.

---

## �🐧 ENVIRONMENT: NATIVE LINUX

**All training happens in native Linux (Ubuntu) environment.**

- Paths: `/home/m_tes/groundthink/...` (Linux syntax)
- Commands: bash/shell (not Windows/PowerShell)
- Virtual env: `source .venv/bin/activate`
- Training script: `python train_v4.py` (runs natively with CUDA kernels)

**Agents: Use Linux syntax for ALL file operations and terminal commands.**

---

## 🔧 Model Registry & Config System (V4.5+)

**As of 2026-01-10, we use a centralized model registry and YAML config system.**

### Quick Start (Copy-Paste Ready)

```bash
# Activate environment
cd /home/m_tes/groundthink && source .venv/bin/activate

# Train with config file (PREFERRED)
python train_v4.py --config configs/train_8m_50k.yaml

# Train with CLI overrides
python train_v4.py --model 8M --max-steps 1000 --lr 0.0003

# Quick test run (200 steps)
python train_v4.py --config configs/train_quick.yaml

# Resume from checkpoint
python train_v4.py --config configs/train_8m_50k.yaml --resume ckpt_HY_step5000.pt
```

### Available Models

Use `--model NAME` to select (or specify in YAML config):

| Name | Params | Description |
|------|--------|-------------|
| `1M` | ~1M | Tiny test model |
| `5M` | ~3.6M | Phase 1 baseline |
| `HY` | ~3.6M | Hybrid v4 (same as 5M) |
| `GF` | ~3.6M | Gated Fusion variant |
| `GF-MH` | ~3.6M | **Phase 2 winner** - Gated Fusion + Mamba-Heavy (70% Mamba) |
| `GF-RH` | ~3.6M | Gated Fusion + RWKV-Heavy (70% RWKV) |
| `8M` | ~7.9M | Scaled model for Phase 3 |

### Available Configs

| File | Purpose |
|------|---------|
| `configs/train_8m_50k.yaml` | 50K step training for 8M model |
| `configs/train_quick.yaml` | Quick test (200 steps, 5M model) |
| `configs/train_default.yaml` | Default 5K step baseline |

### Config Priority (Highest to Lowest)

1. **CLI arguments** - `--max-steps 100` overrides everything
2. **YAML config file** - `--config configs/train_8m_50k.yaml`
3. **DEFAULT_CONFIG** in train_v4.py - fallback values

### Creating New Configs

```yaml
# configs/my_experiment.yaml
model: GF-MH
max_steps: 10000
warmup_steps: 500
batch_size: 32
learning_rate: 0.0003
grad_accum_steps: 4
seq_length: 128
use_amp: true
```

### ⚠️ DO NOT Edit Imports

**Old way (DON'T DO THIS):**
```python
# ❌ WRONG - editing train_v4.py imports
from hybrid_v4_8m import create_hybrid_8m as create_model
```

**New way (CORRECT):**
```bash
# ✅ RIGHT - use CLI argument
python train_v4.py --model 8M
```

The model registry (`models/__init__.py`) handles all imports automatically.

---

## How to Use This Document

1. **Before training**: Review Critical Rules and Pre-Flight Checklist
2. **During training**: Use Quick Reference for log interpretation
3. **When stuck**: Consult Common Failure Modes and Decision Tree
4. **After training**: Grade run using Run Assessment Scorecard (Appendix H)

---

## Critical Rules (Non-Negotiable)

### 1. Validation Loss is THE Metric
- **Train loss means nothing alone** - it only shows the model is learning *something*
- **Val loss determines everything**: when to stop, when to checkpoint, when to adjust
- Log format: `Step X | Train: X.XX | Val: X.XX | RWKV/Mamba: X.XX`
- Eval every 100 steps minimum, every 20 steps during debugging

### 2. Component Gradient Monitoring
Hybrid models can have one component die silently. Track gradient norms separately.

| RWKV/Mamba Ratio | Status | Action |
|------------------|--------|--------|
| 0.3 - 3.0 | OK | Continue |
| 0.1 - 0.3 or 3.0 - 10.0 | WARN | Monitor closely, may need LR adjustment |
| < 0.1 or > 10.0 | FAIL | Stop. One component is dead. |

**Activation Collapse Monitoring** (SSM-specific failure mode):
- SSMs can have "activation collapse" where outputs become near-constant
- Monitor activation variance per component (already logged as `RWKV: var=X.XX` and `Mamba: var=X.XX`)
- **Healthy**: Both variances in 0.5-2.0 range
- **WARN**: Variance < 0.1 or > 5.0 (extreme)
- **FAIL**: Variance ≈ 0 (activations collapsed to constant)

### 3. Stopping Criteria
**STOP training when:**
- Val loss diverges from train loss (gap grows > 0.5)
- Val loss increases for 5+ consecutive evals
- **Val loss increases >5-10% while train decreases** = overfitting, stop immediately
- **Val loss hasn't improved for 3-5x typical improvement interval** (e.g., if improvement every 500 steps, stop after 1500-2500 with no improvement)
- LR reduced 2+ times with no improvement
- Gradient ratio enters FAIL zone
- Loss oscillates wildly (std > 20% of mean)

**CONTINUE training when:**
- Val loss has "heartbeat" (small improvements even if slow)
- Both components have non-zero gradients
- Loss is stable or decreasing

**Hybrid-Specific: Sawtooth Validation Curves**
- Hybrid validation curves often show "sawtooth" pattern (up-down-up-down)
- **Look at the envelope of minima**, not individual points
- If the minima are getting lower over time → training is working
- If the minima have plateaued → model has converged
- If the minima are trending UP → overfitting

### 4. Plateau Response Protocol
| Plateau Duration | Diagnosis | Action |
|-----------------|-----------|--------|
| 1-10% of training | Normal settling | Wait, continue |
| 10-20% of training | Possible issue | Reduce LR by 30-50%, continue 10-20% more steps |
| > 20% of training | Converged or stuck | Stop, analyze |

**Plateau Diagnosis:**
- **Genuine convergence**: Val loss flat but not increasing. Model reached its capacity.
- **Optimization mismatch**: One component's LR is wrong. Check gradient ratio.
- **Gradient competition**: Components fighting each other. Check if updates cancel out.

**For Hybrids at Small Scale (8M):** Investigate each plateau. Don't train through blindly.
- This is different from production scale where "patience budget" applies
- At validation scale, every plateau is information about the architecture

### 5. Defining "Improvement Interval"

The improvement interval is how often you typically see val loss decrease during healthy training.

**How to calculate:**
1. During first 20% of training, track when val_loss improves
2. Average the gap between improvements = your improvement interval
3. Use 3-5x this value as your patience threshold

**Example:**
- Training for 5000 steps, eval every 50 steps
- During steps 0-1000, val improved at: 50, 100, 200, 350, 500, 700
- Average gap: ~100 steps
- Patience threshold: 300-500 steps without improvement

**For our current runs:**
- 5000 steps, eval every 50 steps
- Typical improvement interval: ~100-200 steps during healthy learning
- Patience: stop after 500-1000 steps with no val improvement

---

## Hyperparameters (V4 Defaults)

```python
CONFIG = {
    # Architecture
    'batch_size': 4,
    'seq_len': 256,
    'grad_accum': 4,        # Effective batch = 16
    
    # Optimizer
    'lr': 3e-4,             # Base LR for RWKV + other
    'mamba_lr_mult': 2.0,   # Mamba gets 2x LR (6e-4)
    'min_lr': 3e-5,         # 10% of peak
    'weight_decay': 0.1,
    'betas': (0.9, 0.95),
    'grad_clip': 1.0,
    
    # Schedule
    'warmup_steps': 2000,   # 2-4x longer for hybrids
    'max_steps': 50000,
    
    # Monitoring
    'eval_interval': 100,
    'log_interval': 20,
    'checkpoint_interval': 10000,
    
    # Stopping
    'val_patience': 5,      # Stop after 5 evals with no val improvement
    'grad_ratio_warn': 3.0,
    'grad_ratio_fail': 10.0,
}
```

---

## Scheduler Implementation (V3 Archive Compliance)

**The canonical scheduler pattern from `archive/training.py` and `archive/train_v030.py`:**

```python
# Linear warmup → cosine decay (V3 standard)
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps  # Linear ramp
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay
```

**Current train_v4.py is compliant** with this pattern.

### V3 Parameter Group Structure

From `archive/layers_v030.py` (lines 618-658), V3 used **three parameter groups**:

| Group | LR | Weight Decay | Patterns |
|-------|-----|--------------|----------|
| **decay** | base_lr | 0.1 | Standard weights (projections, FFNs) |
| **recurrent** | base_lr × 0.1 | 0.0 | time_decay, base_decay, gate, grounding |
| **no_decay** | base_lr | 0.0 | bias, norm, ln, A_log, delta, gamma, h0, embed |

**Key insight:** V3 used **lower LR for recurrent/state params** (0.1x multiplier), not higher.

### Warmup Duration Guidelines (from V3_RESEARCH_NOTES.md)

| Source | Recommendation | Our Current |
|--------|----------------|-------------|
| Section 2.15 | Linear warmup 5-10% of total steps | 500/1000 = **50%** ⚠️ |
| V3_CROSS_REFERENCE 1.6 | 2-4x longer warmup for hybrids than pure transformers | Using 500 steps |
| train_v030.py default | warmup=500 for num_steps=5000 (10%) | Matches |

**Note:** Our current 500-step warmup for 1K-step experiments is **50% of training** — significantly higher than the 5-10% guideline. This may be intentional for small-scale validation but should be reduced for longer runs.

### Per-Group Schedules (NEW — Not in V3)

V3 implemented **per-group base LR** but used a **single scheduler** applied uniformly. Task 37 proposes extending this to **per-group warmup durations**:

```python
# V3 approach (single scheduler, different base LRs)
scheduler = LambdaLR(optimizer, lr_lambda)  # Same lambda for all groups

# Task 37 approach (per-group lambdas with different warmups)
scheduler = LambdaLR(optimizer, lr_lambda=[rwkv_lambda, mamba_lambda, other_lambda])
```

**PyTorch supports this natively** — the `lr_lambda` parameter can be a list of functions, one per parameter group.

---

## Per-Component Warmup Schedules (Phase 3.8+)

**Context:** Hybrid models with gated fusion (e.g., GF-MH) show RWKV dominance due to smoother gradients. Task 37 (Phase 3.8) explores **differential warmup schedules** to help Mamba learn independently during early training.

### Archive Findings (V3_RESEARCH_NOTES.md)

| Source | Key Finding |
|--------|-------------|
| Section 2.15/9.6 | "Differential LR: 1.5-3x higher for SSM component initially" |
| Section 2.31 | Recurrent params (gates, decay) get **0.1x** LR in V3 implementation |
| Section 2.8 | "Warm-up 2-4x longer for hybrids than pure transformers" |
| Line 324 | "LR Warmup: Linear warmup for 5-10% of tokens. Prevents Mamba 'A' matrix from exploding" |
| Line 937 | "Per-Group LR: Higher for states, lower for heads" |

**Apparent contradiction:** Research notes recommend higher LR for SSM, but V3 implementation used **lower** LR (0.1x) for recurrent params. This suggests the guidance evolved during development.

### Why Warmup Matters for Hybrids

**RWKV vs Mamba warmup requirements differ fundamentally:**
- **RWKV-6**: Smooth, well-behaved gradients. Standard cosine schedule works well. BlinkDL recommendations: 20-2500 steps warmup depending on model size.
- **Mamba-2**: SSM recurrent dynamics are sensitive to initialization. Requires careful state accumulation. V3 notes: "Prevents Mamba 'A' matrix from exploding before seeing enough data."

**Problem:** Single warmup schedule applies uniformly to all parameter groups, which may not match each component's learning requirements.

**Hypothesis:** If Mamba has a longer/slower warmup period, it develops learned patterns independently before gates "decide" on RWKV dominance.

### Literature Findings (Phase 3.8 Research)

| Component | Warmup Duration | Formula | Source |
|-----------|-----------------|---------|--------|
| RWKV-6 | 20-2500 steps | Linear ramp OR `lr * (0.01 + 0.99 * step/warmup)` (spike fix) | BlinkDL RWKV-LM |
| Mamba-2 | No explicit guidance | Follows GPT-3 (usually 2-5% of total steps) | state-spaces/mamba docs |
| Generic Transformer | 5-10% of total steps | Linear ramp | GPT-3 paper |

**Key Insight:** Hybrid models need **per-group LR schedules**. PyTorch's `LambdaLR` supports this natively via `lr_lambda=[lambda1, lambda2, lambda3]` for multiple parameter groups.

### Implementation Pattern (train_v4.py)

```python
def get_parameter_groups(model, base_lr, mamba_lr_mult=0.5):
    """Return (params_list, lr_list) for per-group warmup."""
    rwkv_params = [p for name, p in model.named_parameters() if 'rwkv' in name.lower()]
    mamba_params = [p for name, p in model.named_parameters() if 'mamba' in name.lower()]
    other_params = [p for name, p in model.named_parameters() 
                    if 'rwkv' not in name.lower() and 'mamba' not in name.lower()]
    
    return (
        [{'params': rwkv_params, 'lr': base_lr},
         {'params': mamba_params, 'lr': base_lr * mamba_lr_mult},
         {'params': other_params, 'lr': base_lr}],
        [base_lr, base_lr * mamba_lr_mult, base_lr]
    )

def get_lr_lambda_per_group(warmup_steps, max_steps, group_warmup_mult=1.0):
    """Return lambda for per-group warmup.
    
    Args:
        warmup_steps: Base warmup duration (e.g., 500)
        max_steps: Total training steps
        group_warmup_mult: Multiplier for this group's warmup (1.0=base, 2.0=2x longer)
    """
    adjusted_warmup = int(warmup_steps * group_warmup_mult)
    
    def lr_lambda(step):
        if step < adjusted_warmup:
            # Linear warmup
            return float(step) / float(max(1, adjusted_warmup))
        else:
            # Cosine decay from adjusted_warmup to max_steps
            progress = float(step - adjusted_warmup) / float(max(1, max_steps - adjusted_warmup))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return lr_lambda

# Usage:
param_groups, base_lrs = get_parameter_groups(model, base_lr=3e-4, mamba_lr_mult=0.5)
optimizer = AdamW(param_groups, weight_decay=0.1)

# Create per-group warmup schedules
rwkv_lambda = get_lr_lambda_per_group(warmup_steps=500, max_steps=5000, group_warmup_mult=1.0)
mamba_lambda = get_lr_lambda_per_group(warmup_steps=500, max_steps=5000, group_warmup_mult=2.0)  # 2x longer
other_lambda = get_lr_lambda_per_group(warmup_steps=500, max_steps=5000, group_warmup_mult=1.0)

scheduler = LambdaLR(optimizer, lr_lambda=[rwkv_lambda, mamba_lambda, other_lambda])
```

### Experiments (Phase 3.8 Task 37)

**37a: Mamba Extended Warmup**
- RWKV: standard 500-step warmup
- Mamba: 1000-step warmup (2x longer)
- Hypothesis: Slower Mamba learning early on reduces gate collapse
- Expected: R/M ratio improvement (0.10 → 0.3+), slight val loss trade-off

**37b: RWKV Slow Ramp (BlinkDL Spike-Fix)**
- RWKV: `lr * (0.01 + 0.99 * step/warmup_steps)` for first 500 steps
- Mamba: standard linear warmup
- Hypothesis: RWKV smooth ramp prevents gradient spikes that drown out Mamba
- Expected: More stable learning trajectory, possible balance improvement

**37c: Mamba Delayed Start**
- Mamba: Zero LR for first 250 steps, then standard warmup
- RWKV: standard warmup
- Hypothesis: Prevents Mamba from immediately losing to smoother RWKV
- Expected: R/M ratio improvement if pure gate dominance is the issue

**37d: Combined (Slow RWKV + Extended Mamba)**
- RWKV: Slow ramp formula (spike prevention)
- Mamba: Extended warmup (2x longer)
- Hypothesis: Attack signal dominance from both angles
- Expected: Best balance improvement if both factors contribute

### Evaluation Criteria

For each experiment:
1. **R/M Ratio Trajectory**: Plot gradient ratio over training
   - Success: Ratio stays in 0.3-0.5 range for majority of training
   - Failure: Ratio drops to <0.1 by step 500 (gate collapse)

2. **Val Loss**: Should not degrade significantly
   - Acceptable: < 5% worse than baseline GF-MH (1.59 baseline)
   - Good: Within 2% of baseline
   - Success: Improved relative to baseline

3. **Component Gradient Magnitude**:
   - Mamba gradients should remain non-zero throughout
   - RWKV gradients should stabilize earlier (better conditioned)

### Decision Logic (If Implementing)

```python
# After each experiment, log:
# - Final R/M ratio
# - Val loss (normalized vs baseline)
# - Warmup-phase learning curves (steps 0-500)

if ratio_improved and val_loss_acceptable:
    # Adopt this schedule
    update_train_config(schedule_params)
elif ratio_improved and val_loss_worse:
    # Consider hybrid: use schedule for first N steps, then revert
    implement_staged_warmup(schedule_params, revert_step=1000)
else:
    # Stick with current approach or try next variant
    proceed_to_37b()
```

---

## Training Loop Structure

```python
# Pseudocode - actual implementation in train_v4.py

for step in range(max_steps):
    # 1. Forward + backward (with gradient accumulation)
    loss = forward_backward(batch)
    
    # 2. Gradient monitoring (BEFORE clipping)
    rwkv_norm, mamba_norm, ratio = get_component_gradients(model)
    check_gradient_health(ratio)  # Warn/fail if bad
    
    # 3. Clip + step
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    
    # 4. Validation (every eval_interval steps)
    if step % eval_interval == 0:
        val_loss = evaluate(model, val_data)
        log(step, train_loss, val_loss, ratio)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, 'best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= val_patience:
                print("Early stopping: val loss not improving")
                break
```

---

## Common Failure Modes

### 1. "Loss looks good but model outputs garbage"
- **Cause**: Overfitting. Train loss ↓ but val loss ↑
- **Fix**: More data, more dropout, earlier stopping

### 2. "One component dominates"
- **Symptoms**: Gradient ratio >> 10 or << 0.1
- **Cause**: LR imbalance or architecture issue
- **Fix**: Adjust mamba_lr_mult, check layer initialization

### 3. "Loss stuck from the start"
- **Cause**: LR too low, or dead initialization
- **Fix**: Try 10x higher LR, check that all params have grads

### 4. "Loss spikes randomly"
- **Cause**: Grad explosion, bad batch, or numerical instability
- **Fix**: Lower LR, increase grad_clip, check for inf/nan

### 5. "Training is stable but val never improves"
- **Cause**: Dataset too small, or model too big
- **Fix**: More data, or smaller model, or stronger regularization

### 6. "Activation Collapse" (SSM-specific)
- **Symptoms**: One component's activation variance → 0, outputs become constant
- **Cause**: State dynamics collapsed, often from bad initialization or gradient starvation
- **Diagnosis**: Look at logged `RWKV: var=X.XX` and `Mamba: var=X.XX` values
- **Fix**: Check initialization, increase that component's LR, or reduce the other component's dominance

### 7. "Components Fighting" (Gradient Competition)
- **Symptoms**: Loss oscillates, gradient ratio swings between extremes
- **Cause**: RWKV and Mamba optimizing for different local minima
- **Diagnosis**: Gradient ratio alternates high/low across steps
- **Fix**: Reduce LR of dominant component, or add separate normalization layers

---

## Checkpointing Strategy

1. **Best checkpoint**: Save whenever val_loss improves
2. **Periodic checkpoint**: Every 10K steps regardless
3. **Always save**: optimizer state + scheduler state + step count

```python
torch.save({
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'config': CONFIG,
}, f'checkpoint_step_{step}.pt')
```

---

## Pre-Flight Checklist

Before starting a real training run:

- [ ] Ran 100-step test, loss decreasing
- [ ] Val loss computed and logged
- [ ] Component gradients both non-zero
- [ ] Gradient ratio in OK range (0.3-3.0)
- [ ] Checkpoint saving works
- [ ] VRAM usage acceptable (< 80% of available)
- [ ] Estimated time to completion calculated

---

## Quick Reference: Log Interpretation

```
Step  100 | Train: 4.50 | Val: 4.52 | RWKV/Mamba: 1.20       # Good: val ≈ train
Step  200 | Train: 4.30 | Val: 4.35 | RWKV/Mamba: 0.95       # Good: both improving
Step  300 | Train: 4.10 | Val: 4.40 | RWKV/Mamba: 0.80       # WARN: gap growing
Step  400 | Train: 3.90 | Val: 4.50 | RWKV/Mamba: 0.70       # BAD: overfitting
Step  500 | Train: 3.70 | Val: 4.60 | RWKV/Mamba: 8.50 [WARN] # BAD: ratio + overfit
```

**Healthy training**: Train and val decrease together, ratio stable 0.5-2.0

---

## Files Reference

| File | Purpose |
|------|---------|
| `train_v4.py` | Training script with all monitoring |
| `hybrid_v4.py` | Model architecture (RWKV6 + Mamba2 parallel) |
| `data_v030.py` | Dataset with train/val split |
| `tokenizer_v030.py` | Character tokenizer |
| `V4_HANDOFF.md` | Task tracking for agents |
| `V4_DESIGN.md` | Architecture decisions |
| `V4_BUILD_LOG.md` | Session-by-session progress |

---

*Last updated: 2026-01-09*

---

## Appendix A: Weight Decay (NOT YET IMPLEMENTED)

**Current state:** train_v4.py applies same weight_decay to all params in each group.

**TODO:** Should exclude these patterns from weight decay:
```python
no_decay = ['bias', 'norm', 'ln', 'gain', 'scale']
```

FLA internal params (A_log, time_decay) are handled by FLA - we don't control their weight decay directly.

### Fusion Gains (V4 Approach)

V4 uses `rwkv_gain` and `mamba_gain` initialized to 0.5 each:
```python
self.rwkv_gain = nn.Parameter(torch.ones(hidden_size) * 0.5)
self.mamba_gain = nn.Parameter(torch.ones(hidden_size) * 0.5)
```

This is different from the 0.01 residual scaling - V4's parallel design already provides gradient independence, so we start with equal contribution from both pathways.

---

## Appendix B: V4 Architecture

### ParallelHybridBlock

Each block runs RWKV-6 and Mamba-2 **in parallel** on the same input:

```
x_in ─→ RMSNorm ─┬─→ [RWKV6Attention] ─→ * rwkv_gain ─┐
                 │                                     │
                 └─→ [Mamba2]          ─→ * mamba_gain ┴─→ + x_in ─→ RMSNorm ─→ FFN ─→ out
```

### Key Parameters

| Component | Setting | Notes |
|-----------|---------|-------|
| `RWKV6Attention` | hidden_size, num_heads | From FLA library |
| `Mamba2` | expand=2, head_dim=64 | num_heads = expand*hidden/head_dim |
| `rwkv_gain` | init 0.5 | Learnable per-channel |
| `mamba_gain` | init 0.5 | Learnable per-channel |
| `FFN` | 4x expansion, GELU | Standard transformer FFN |

### Model Configs

| Name | hidden_size | num_layers | num_heads | Params |
|------|-------------|------------|-----------|--------|
| `create_hybrid_5m` | 128 | 8 | 4 | ~3M |
| `create_hybrid_8m` | 192 | 12 | 6 | ~8M |

### Weight Initialization
- Linear: `normal_(mean=0.0, std=0.02)`
- Embedding: `normal_(mean=0.0, std=0.02)`
- Tied embeddings: `head.weight = embed.weight`

---

## Appendix C: State Monitoring (ADVANCED)

### FLA Internal States

FLA's RWKV6Attention and Mamba2 manage their own internal states. We don't have direct access to monitor state entropy.

**What we CAN monitor:**
- Component gradient norms (implemented in train_v4.py)
- Loss trajectory
- Output distribution entropy

**Passkey test:** Could be implemented as a separate evaluation script, but not critical for initial training validation.

---

## Appendix D: Curriculum Learning (FUTURE)

Not implemented in current train_v4.py. For initial validation, fixed seq_len=256 is fine.

If implementing later:
| Phase | Seq Len | Purpose |
|-------|---------|---------|
| Warmup | 128 | Basic patterns |
| Short | 256 | Local dependencies |
| Medium | 512 | State compression |
| Full | 1024 | Long-range |

**Transition rule:** When doubling seq_len, reduce LR by 10-15%.

---

## Appendix E: Evaluation Tests (Reference)

Simple tests to validate model behavior. Can be run manually after training.

### Test 1: Completion Coherence
Feed first 100 chars of shakespeare.txt, generate 100 more. Output should be readable English-like text, not random chars.

### Test 2: Long Context
Feed 500+ tokens, check that loss on token 500 is similar to loss on token 50. Big spike = state is leaking.

### Test 3: Character Distribution
After training, sample 1000 characters. Distribution should roughly match shakespeare (mostly lowercase letters, spaces, punctuation). If output is 90% one character = model collapsed.

### For Conversational Data (Future)
- Persona lock test
- State jitter (cross-document leakage)
- Instruction following

---

## Appendix F: Model Configs & Scaling Rules

### Current Model Configurations

| Config Name | hidden_size | num_layers | num_heads | Approx Params | Use Case |
|-------------|-------------|------------|-----------|---------------|----------|
| `create_hybrid_1m` | 64 | 4 | 2 | ~500K | Ultra-fast iteration, sanity checks |
| `create_hybrid_5m` | 128 | 8 | 4 | ~3M | **Current validation** |
| `create_hybrid_8m` | 192 | 12 | 6 | ~8M | Extended validation |
| (future) | 256 | 12 | 8 | ~15M | Pre-scale testing |
| (future) | 384 | 12 | 12 | ~30M | First real scale |

### Scaling Strategy

**Rule 1:** Scale width first, then depth. Changing depth invalidates tuning.

**Rule 2:** When doubling width:
- Expect ~4x params (width² in attention/FFN)
- May need to reduce LR by ~30%
- Warmup should increase proportionally

**TODO:** Add `create_hybrid_1m` for faster iteration testing:
```python
def create_hybrid_1m(vocab_size: int = 10000) -> HybridModel:
    """Create ~500K parameter hybrid model for fast iteration"""
    return HybridModel(
        vocab_size=vocab_size,
        hidden_size=64,
        num_layers=4,
        num_heads=2,
        ffn_mult=4.0,
    )
```

### Expected Training Times (Windows/Triton fallback @ ~2600 tok/s)

| Model | 5000 steps | 10000 steps |
|-------|------------|-------------|
| 1M (TODO) | ~5 min | ~10 min |
| 3M (current) | ~30 min | ~60 min |
| 8M | ~60 min | ~120 min |

**Note:** On Linux with proper CUDA kernels, expect 10-20x faster.

---

## Appendix G: Tokenizer (CURRENT STATE)

Using char-level tokenizer (~97 vocab for shakespeare.txt).

- ✅ Minimal embedding tax
- ✅ Fast iteration
- ❌ Not suitable for production (need BPE for real deployment)

For scaling to larger data, switch to custom BPE (16-24k vocab).

---

## Quick Decision Tree

```
Loss stuck at 7.0?
├── Check gradient norm
│   ├── Near 0 → Vanishing gradients, raise LR
│   └── Spiking → Lower LR, add grad clip
├── Check component gradients
│   └── One near zero → That pathway is dead
└── Check LR
    └── Try 10x higher or lower

Val loss diverging from train?
├── Gap < 0.2 → Normal, continue
├── Gap 0.2-0.5 → Watch closely
└── Gap > 0.5 → Overfitting, stop or regularize

Component gradient ratio unhealthy?
├── Ratio < 0.1 or > 10 → FAIL, architecture problem
├── Ratio 0.1-0.3 or 3-10 → WARN, adjust mamba_lr_mult
└── Ratio 0.3-3.0 → Healthy, continue
```

---

## Appendix H: Run Assessment Scorecard

Use this to grade any training run. Each criterion is Pass/Warn/Fail.

### Criterion 1: Validation Loss Trend
| Grade | Condition |
|-------|-----------|
| ✅ PASS | Val loss decreased over run, final < initial |
| ⚠️ WARN | Val loss flat or sawtooth but envelope improving |
| ❌ FAIL | Val loss increased or diverged from train |

### Criterion 2: Train/Val Gap
| Grade | Condition |
|-------|-----------|
| ✅ PASS | Gap < 0.2 throughout |
| ⚠️ WARN | Gap 0.2-0.5, stable |
| ❌ FAIL | Gap > 0.5 or growing |

### Criterion 3: Gradient Ratio Stability
| Grade | Condition |
|-------|-----------|
| ✅ PASS | Ratio in [0.3, 3.0] for >80% of evals |
| ⚠️ WARN | Ratio in [0.1, 10.0] but outside [0.3, 3.0] frequently |
| ❌ FAIL | Ratio < 0.1 or > 10.0 at any point |

### Criterion 4: Activation Variance
| Grade | Condition |
|-------|-----------|
| ✅ PASS | Both components in [0.5, 2.0] range |
| ⚠️ WARN | One component outside [0.3, 3.0] |
| ❌ FAIL | Either component < 0.1 (collapsed) |

### Criterion 5: Improvement Interval
| Grade | Condition |
|-------|-----------|
| ✅ PASS | Val improves within 3x typical interval |
| ⚠️ WARN | Plateau 10-20% of training |
| ❌ FAIL | No improvement for >20% of training |

### Overall Grade
- **A (Excellent)**: 5 PASS
- **B (Good)**: 4 PASS, 1 WARN
- **C (Acceptable)**: 3 PASS, 2 WARN
- **D (Marginal)**: 2 PASS, 3 WARN
- **F (Failed)**: Any FAIL

---

## Appendix I: Example Run Assessment

### HY Fusion Run (2026-01-09) - Steps 1000-2000

**Run Info:**
- Model: V4 HY Fusion (~3M params)
- Dataset: shakespeare.txt (1.1M chars, 97 vocab)
- Config: batch=8, seq=256, lr=3e-4, mamba_lr=6e-4

**Observations:**

| Step | Train PPL | Val PPL | R/M Ratio | Status |
|------|-----------|---------|-----------|--------|
| 1050 | 4.26 | 4.62 | 3.32 | WARN |
| 1500 | 4.08 | 4.15 | 2.85 | OK |
| 2000 | 3.73 | 3.94 | 2.64 | OK |

**Scorecard:**

| Criterion | Grade | Notes |
|-----------|-------|-------|
| 1. Val Loss Trend | ✅ PASS | 4.62 → 3.94 (15% improvement) |
| 2. Train/Val Gap | ✅ PASS | Gap = 0.21 at step 2000 |
| 3. Gradient Ratio | ⚠️ WARN | ~50% in OK range, ~50% WARN (3.0-3.5) |
| 4. Activation Var | ✅ PASS | RWKV=0.87, Mamba=1.08 (healthy) |
| 5. Improvement Int | ✅ PASS | Improving every 100-200 steps |

**Overall: B (Good)** - 4 PASS, 1 WARN

**Recommendation:** Continue training. Gradient ratio oscillating near threshold is not blocking - trend is toward balance. Monitor for stabilization.

---

## Appendix J: Training Summary Output (TODO)

**Future Enhancement:** Add `--save-summary` flag to train_v4.py that outputs a markdown summary after training:

```markdown
# Training Summary: [RUN_NAME]

## Run Configuration
- **Model:** V4 HY Fusion (3,092,736 params)
- **Dataset:** shakespeare.txt (1,115,394 tokens)
- **Date:** 2026-01-09 14:30
- **Duration:** 32 minutes
- **Steps:** 1000 → 2000

## Final Metrics
- Train Loss: 1.32 (PPL 3.73)
- Val Loss: 1.37 (PPL 3.94)
- Best Val Loss: 1.37 (step 2000)

## Health Summary
- Gradient Ratio: 2.64 (OK)
- RWKV Variance: 0.87 (healthy)
- Mamba Variance: 1.08 (healthy)
- Train/Val Gap: 0.21 (healthy)

## Scorecard: B (Good)
[auto-generated scorecard]

## Checkpoints
- ckpt_HY_step1000.pt (resumed from)
- ckpt_HY_step2000.pt (saved)
```

**Implementation Notes:**
- Save to `runs/[RUN_NAME]_summary.md`
- Include version of train_v4.py and hybrid_v4.py
- Include git hash if available
