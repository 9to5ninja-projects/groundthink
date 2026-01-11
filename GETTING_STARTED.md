# Getting Started with GroundThink

**Version:** V0.5 Phase 0 — Base Model Characterization  
**V4 Status:** ✅ Graduated  
**Goal:** Set up environment for benchmarking pure RWKV-6 and Mamba-2.

See [ABOUT.md](ABOUT.md) for project overview and [BASE_MODEL_CHARACTERIZATION.md](BASE_MODEL_CHARACTERIZATION.md) for current research plan.

---

## Hardware Requirements

| Scale | Min VRAM | Recommended GPU | Notes |
|-------|----------|-----------------|-------|
| 3.5M (Tiny) | 4 GB | RTX 4050, 3060 | Development/testing |
| 8M (Small) | 6 GB | RTX 4060, 3070 | V5 experiments |
| 30M (Medium) | 12 GB | RTX 4080, 3080 | V5 validation |
| 125M (Large) | 24 GB | RTX 3090, A6000 | Full training |

**CUDA Compute Capability:** ≥7.0 (Volta or newer)

---

## Software Requirements

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10+ | `python --version` |
| CUDA | 12.1+ | `nvidia-smi` |
| GCC | 11+ | `gcc --version` |
| Linux | Ubuntu 22.04+ | Native or WSL2 |

**Windows:** Not supported for training (CUDA kernel JIT fails). Use WSL2.

---

## Quick Start

```bash
# 1. Clone and enter directory
git clone --recursive https://github.com/9to5ninja/groundthink.git
cd groundthink

# 2. Create and activate venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install CUDA kernels (required for GPU)
pip install mamba-ssm causal-conv1d

# 5. Run the test suite
python -m pytest tests/test_tiny_graduation.py -v
```

---

## What the Tests Verify

| Test Group | What It Checks |
|------------|----------------|
| `--states` | State accumulation (S0-S4) works correctly |
| `--gates` | G1-G4 gate values are in expected ranges |
| `--overfit` | Model can memorize a micro-batch |
| `--baseline` | train_v4.py runs without crash |
| `--checkpoint` | Resume from checkpoint produces identical state |

Run specific groups:
```bash
pytest tests/test_tiny_graduation.py --gates -v   # Just gate tests
pytest tests/test_tiny_graduation.py -v           # All tests
```

---

## Expected Output

```
tests/test_tiny_graduation.py::test_G1_gate_produces_nonzero_output PASSED
tests/test_tiny_graduation.py::test_G2_gate_values_in_expected_range PASSED
tests/test_tiny_graduation.py::test_G3_gate_gradient_flows PASSED
tests/test_tiny_graduation.py::test_G4_gate_behavior_changes_with_input PASSED
tests/test_tiny_graduation.py::test_baseline_training_completes PASSED
...
```

---

## Key Files

| File | Purpose |
|------|---------|
| [model.py](model.py) | GF-MH model definition |
| [layers.py](layers.py) | RWKV6 + Mamba2 layer implementations |
| [cuda_backends.py](cuda_backends.py) | CUDA kernel imports (our wrapper) |
| [train_v4.py](train_v4.py) | Training loop |
| [configs/](configs/) | YAML config files |

---

## Troubleshooting

### "CUDA not found"
```bash
nvidia-smi  # Should show your GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### "ninja: build stopped: subcommand failed"
- Check GCC version: `gcc --version` (need 11+)
- Update: `sudo apt update && sudo apt install gcc-11 g++-11`

### "mamba_ssm not found"
```bash
pip install mamba-ssm causal-conv1d
```

### Tests hang or segfault
- Check VRAM: `nvidia-smi` (is it full?)
- Reduce batch size in test config

---

## Next Steps

| Goal | Read |
|------|------|
| Understand the project | [ONBOARDING.md](ONBOARDING.md) |
| See architecture details | [V4_DESIGN.md](V4_DESIGN.md) |
| Run benchmarks | [README.md](README.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |
| V5 planning | [V5_GATING.md](V5_GATING.md) |

---

**Test runtime:** ~30 seconds (RTX 4050)  
**Understanding time:** ~1 hour (ONBOARDING + first test)
