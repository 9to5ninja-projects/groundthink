# Agent Handoff

**Version:** 0.5.1.5-Alpha | **Phase:** 1 Task 0.4 | **Updated:** 2026-01-15

---

## Current Status

✅ **Phase 0 COMPLETE** — Base model characterization done.  
✅ **Task 0.1 COMPLETE** — minGRU Arbiter with RMSNorm (gold standard).  
✅ **Task 0.2 COMPLETE** — Mamba Residual Path stable through layers.  
✅ **Task 0.3b COMPLETE** — 8-Layer stability proven.  
✅ **Task 0.3 COMPLETE** — Twin Debate Loss implemented.  
🔧 **Task 0.4 NEXT** — 4M Pilot Run with full modules.

| Test | Result |
|------|--------|
| Task 0.1 minGRU Arbiter | ✅ 99.2% trainable, O(log N) parallel |
| Task 0.2 Mamba Residual | ✅ Stable through 8 layers |
| Task 0.3b 8-Layer Proof | ✅ Mamba >5% at all depths |
| Task 0.3 Debate Loss | ✅ Diversity + Arbiter loss working |

---

## Last Session (2026-01-15)

**Task 0.3 Results:**
- DiversityLoss: Penalizes cosine similarity between agencies
- ArbiterLoss: Rewards trusting better-performing pathway
- TwinDebateLoss: Combined with λ_div=0.1, λ_arb=0.1
- Integration test passed
- Exported to `tools/debate_loss.py`

---

## Next Actions

### Immediate (Task 0.4)
| Priority | Task | Description | Status |
|----------|------|-------------|--------|
| **0.4** | 4M Pilot Run | 5K steps, real modules, Mamba >5% | 🔧 NEXT |

### Phase 1 Graduation Criteria
- [ ] Mamba contribution > 5% (measured by ablation)
- [ ] Variance amplification 2-6x (SSM range)
- [ ] Softmax entropy > 5.0, max_prob < 0.2
- [ ] Arbiter α varies across sequence
- [ ] Loss converges with debate loss enabled

---

## Completed Notebooks

| Notebook | Task | Status |
|----------|------|--------|
| `task_0_0_1_wsl.ipynb` | RWKV-6 baseline | ✅ |
| `task_0_0_2_mamba.ipynb` | Mamba-2 baseline | ✅ |
| `task_0_0_3_gpt1.ipynb` | GPT-1 baseline | ✅ |
| `task_0_1_exploration.ipynb` | GRU exploration | ✅ |
| `task_0_1_v1_glu_baseline.ipynb` | GLU baseline | ✅ |
| `task_0_1b_mingru_comparison.ipynb` | minGRU gold standard | ✅ |
| `task_0_2_mamba_residual_path.ipynb` | Mamba residual | ✅ |
| `task_0_3b_8layer_stability.ipynb` | 8-layer proof | ✅ |
| `task_0_3_debate_loss.ipynb` | Debate loss | ✅ |

---

## Architecture Summary

```
GroundThink 4M Model:
├── Embedding (vocab → d_model)
├── TwinDebateBlock × 8
│   ├── RWKV6TimeMix (amplifier)
│   ├── Mamba2TimeMix + Residual (damper, grounded)
│   ├── minGRUArbiter (RMSNorm → scan → weights)
│   └── Post-norm + Skip
├── LM Head (d_model → vocab)
└── TwinDebateLoss (CE + Diversity + Arbiter)
```

**Target Config:**
- d_model: 256-384 (to hit ~4M params)
- n_layers: 8
- vocab: 16K BPE (WikiText-103)

---

## Key Files

| File | Purpose |
|------|---------|
| `ops/arbiter_mingru.py` | Production minGRU arbiter |
| `tools/debate_loss.py` | Twin Debate loss functions |
| `ops/rwkv6_prototype.py` | RWKV-6 time-mixing |
| `ops/mamba2_prototype.py` | Mamba-2 time-mixing |

---

*For detailed task definitions, see [V0.5_ROADMAP.md](V0.5_ROADMAP.md)*
