# RWKV Training Notes

**Source**: Official BlinkDL/RWKV-LM repository (https://github.com/BlinkDL/RWKV-LM)  
**Retrieved**: 2026-01-12  
**Purpose**: Explain observed softmax saturation in Task 0.0.1 and document correct training practices

## Summary

Our Task 0.0.1 baseline showed:
- Logits range: [-55, +83] (should be much smaller)
- Max probability: 1.0 (saturating)
- Entropy: 1.70 (random = 9.68)

**Root Cause**: Initialization differs significantly from BlinkDL's recommendations.

### ✅ ABLATION CONFIRMED (2026-01-12)

Sub-task 0.0.1.a tested BlinkDL init vs our original init:

| Metric | Original | BlinkDL | Change |
|--------|----------|---------|--------|
| Final loss | 34.3 | 7.9 | **4.3x better** |
| Max logit | 83 | 6.4 | 13x smaller |
| Max prob | 1.0 | 0.082 | No saturation |
| Entropy | 1.7 | 9.2 | Near random |
| Saturation | 15.6% | **0%** | Fixed |

**Conclusion:** Adopt BlinkDL initialization for all future RWKV work.

---

## Official BlinkDL Initialization (RWKV-5/6)

From `generate_init_weight()` in `src/model.py`:

```python
# Embedding - VERY SMALL
emb.weight => nn.init.uniform_(a=-1e-4, b=1e-4)
# Note: ln0 of block0 is the layernorm for emb.weight

# Head (output projection)
head.weight => nn.init.orthogonal_(gain=0.5*sqrt(n_vocab / n_embd))

# Attention weights
att.receptance.weight => nn.init.orthogonal_(gain=1)
att.key.weight => nn.init.orthogonal_(gain=0.1)  # LOW GAIN
att.value.weight => nn.init.orthogonal_(gain=1)
att.gate.weight => nn.init.orthogonal_(gain=0.1)  # LOW GAIN
att.output.weight => zero  # ZERO INIT

# GroupNorm layer scaling (ln_x)
att.ln_x.weight => ((1 + layer_id) / total_layers) ** 0.7

# FFN weights
ffn.key.weight => nn.init.orthogonal_(gain=1)
ffn.value.weight => zero   # ZERO INIT
ffn.receptance.weight => zero  # ZERO INIT
```

---

## Comparison: Ours vs BlinkDL

| Parameter | Our Current | BlinkDL Official | Impact |
|-----------|-------------|------------------|--------|
| `emb.weight` | Default (~N(0,1)) | uniform(-1e-4, 1e-4) | **10,000x too large** |
| `head.weight` | Tied to embed | orthogonal(0.5*sqrt(V/D)) | Tied is fine, but embed too large |
| `ffn.value.weight` | xavier(0.5) | **ZERO** | Residual starts non-zero |
| `att.output.weight` | Default | **ZERO** | Residual starts non-zero |
| `att.key.weight` | Default | orthogonal(0.1) | Key scaling matters |
| Weight decay | 0.1 to all params | 0.1 only to projections | LN/bias shouldn't decay |
| Normalization | RMSNorm | PreLN LayerNorm | Minor difference |

---

## Fixing RWKV-6 Training Spikes

From BlinkDL's README:

1. **Upgrade to RWKV-7** (most stable, spike-free)
2. **K-clamping**: Add `k = k * torch.clamp(w, max=0).exp()` before WKV kernel
3. **Adam eps**: Use `--adam_eps 1e-18`
4. **Beta2**: Use `--beta2 0.95` if seeing spikes
5. **Warmup**: `lr = lr * (0.01 + 0.99 * step / warmup_steps)` with `--warmup_steps 20`
6. **Weight decay**: 0.1 with `lr_final = lr_init / 100`

---

## Small Model / Small Data Tips

Direct quote from BlinkDL:

> "When I train RWKV music models, I use deep & narrow (such as L29-D512) dimensions, and apply wd and dropout (such as wd=2 dropout=0.02). Note RWKV-LM dropout is very effective - use 1/4 of your usual value."

**Implications for our 8L×144D model:**
- Consider deeper architecture (more layers, smaller hidden)
- Apply dropout (0.02) with `x = x + dropout(att(x))`
- Weight decay can be higher (up to 2.0)

---

## Critical: Weight Decay Application

Direct quote from BlinkDL:

> "Only apply weight decay to large matrix parameters (basically projections) in your model instead of all parameters. THIS IS VERY IMPORTANT."

**Correct implementation:**
```python
def get_parameter_groups(model, base_lr=3e-4, wd=0.1):
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if p.dim() >= 2:  # Matrix weights
            decay_params.append(p)
        else:  # Biases, LayerNorm, small params
            no_decay_params.append(p)
    return [
        {'params': decay_params, 'weight_decay': wd},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
```

---

## PreLN LayerNorm vs RMSNorm

Direct quote from BlinkDL:

> "Use PreLN LayerNorm (instead of RMSNorm) for RWKV. I think it's related to better initial state, because I am not using trainable initial state (found it useless when using LayerNorm)."

---

## Action Items for v0.5.0.1

### Immediate (Task 0.0.1 Fix):
1. [ ] Add initialization ablation cell to notebook
2. [ ] Test tiny embedding init: `uniform(-1e-4, 1e-4)`
3. [ ] Test zero-init for FFN output and att.output
4. [ ] Apply weight decay only to projections

### Future (Hybrid Training):
1. [ ] Implement proper BlinkDL initialization in production model
2. [ ] Consider PreLN LayerNorm instead of RMSNorm
3. [ ] Add K-clamping for RWKV-6 stability
4. [ ] Test deeper/narrower architectures for small models

---

## References

- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) - Official repository
- [RWKV-5/6 Paper](https://arxiv.org/abs/2404.05892) - Eagle/Finch paper
- [rwkv7_train_simplified.py](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/rwkv7_train_simplified.py) - Reference implementation

---

## Observation Log

### 2026-01-12: Task 0.0.1 Softmax Saturation Explained

**Observed**:
- Logits: [-55, +83] (exploding)
- Max prob: 1.0 (saturating)
- Entropy: 1.70 vs random 9.68

**Root Cause Analysis**:
1. Embedding init too large (default ~N(0,1) vs required uniform(-1e-4, 1e-4))
2. FFN output not zero-initialized (residual stream starts non-zero)
3. Weight decay applied to LayerNorm (causes drift)

**Conclusion**: Softmax saturation is NOT an RWKV-6 architectural flaw. It's an initialization/training configuration issue that BlinkDL has solved. Our AMPLIFIER characterization (1.28x/layer) is valid, but the saturation can be mitigated with proper initialization.

**Next Steps**: Create ablation cell to verify these hypotheses before proceeding to Task 0.0.2.
