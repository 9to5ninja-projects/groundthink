# Getting Started with GroundThink in 5 Minutes

**Goal:** Verify your environment and run your first benchmark.

---

## Prerequisites
- **OS:** Linux/WSL (native required; see [README.md](README.md))
- **Python:** 3.10+
- **CUDA:** 12.1+ (for GPU acceleration)
- **Disk:** ~5GB free (for deps + models + checkpoints)

---

## Quick Start (Copy-Paste)

```bash
# 1. Clone and enter directory
git clone https://github.com/9to5ninja-projects/groundthink.git
cd groundthink

# 2. Setup environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Optional: Faster Mamba kernels (Linux only)
pip install causal-conv1d mamba-ssm

# 4. Verify everything works
python test_phase0_complete.py

# 5. Run your first benchmark (takes ~20 min on A100)
python benchmark_variants.py
```

---

## What Happened?

✅ **test_phase0_complete.py** verified:
- CUDA detection
- PyTorch installation
- FLA library working
- Model forward pass (no crashes)

✅ **benchmark_variants.py** ran all 7 Phase 2 variants:
- Trained each for 500 steps
- Logged loss, throughput, validation metrics
- Compared results in a summary table

---

## Expected Output

You should see something like:

```
============ BENCHMARK RESULTS ============

Variant | Val Loss | Train Loss | Throughput
--------|----------|------------|------------
GF      | 1.6891   | 1.6536     | 42.9K tok/s
CP      | 1.6919   | 1.6544     | 47.7K tok/s
HY      | 1.7600   | 1.7289     | 31.7K tok/s
... (4 more)

🏆 WINNER: GF-MH (Gated Fusion + Mamba-Heavy)
Val Loss: 1.6700 (-1.8% vs balanced GF)

========================================
```

---

## Next Steps

### 👀 Just Want to Understand the Project?
1. Read [ONBOARDING.md](ONBOARDING.md) Part 1-4 (~15 min)
2. Look at results above
3. Read [README.md](README.md) "Phase 2 Results" section

### 🔬 Want to Add a New Variant?
1. Copy `hybrid_v4_GF.py` to `hybrid_v4_MY.py`
2. Modify the fusion mechanism in the `forward()` method
3. Add your variant to `benchmark_variants.py` in the `variants` dict
4. Run: `python benchmark_variants.py` to see how it compares
5. See [CONTRIBUTING.md](CONTRIBUTING.md) for full instructions

### 🚂 Want to Train Longer?
1. Open `train_v4.py` or `train.py`
2. Modify `max_steps` (currently 5000)
3. Run: `python train_v4.py` (tracks loss in real-time)
4. Checkpoints save to `logs/` and `checkpoints/`

### 🤔 Something's Broken?
1. Check [V4_BUILD_LOG.md](V4_BUILD_LOG.md) for known issues
2. Check [V4_HANDOFF.md](V4_HANDOFF.md) "AUDIT SUMMARY" for status
3. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) (if it exists) or open an issue

---

## Key Files for Getting Oriented

| File | Purpose | Why Read |
|------|---------|----------|
| [ONBOARDING.md](ONBOARDING.md) | Big picture + concepts | Understand RWKV/Mamba |
| [README.md](README.md) | Quick reference | Phase 2 results + running benchmarks |
| [V4_DESIGN.md](V4_DESIGN.md) | Architecture details | See actual implementation |
| [hybrid_v4_GF.py](hybrid_v4_GF.py) | Code example | Understand the model |
| [benchmark_variants.py](benchmark_variants.py) | Testing framework | How to benchmark fairly |

---

## Troubleshooting

### "CUDA not found"
- Check: `nvidia-smi` (do you see your GPU?)
- Reinstall: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### "FLA module not found"
- Check: `pip list | grep fla`
- Reinstall: `pip install fla-core` (or from source in [ONBOARDING.md](ONBOARDING.md) Part 8)

### "test_phase0_complete.py hangs"
- Kill it (Ctrl+C)
- Check environment: `python -c "import torch; print(torch.cuda.is_available())"`
- See [V4_BUILD_LOG.md](V4_BUILD_LOG.md) Session 1 for setup details

### "Import errors in hybrid_v4_*.py"
- Likely cause: FLA library structure changed
- Fix: Check [V4_HANDOFF.md AUDIT SUMMARY](V4_HANDOFF.md#audit-summary) for known issues
- Reference: See `fla_replacements.py` for custom wrapper

---

## What's Next?

👉 After running the benchmark, explore:
- **Phase 2 Results:** Why did GF-MH win? See [CHANGELOG.md](CHANGELOG.md)
- **Phase 3 Planning:** What's next? See [V4_STRATEGY.md](V4_STRATEGY.md) Phase 3 section
- **Contributing:** Want to add your own variant? See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Still stuck?** Check:
1. [ONBOARDING.md](ONBOARDING.md) Part 10 "Librarian's Note"
2. [V4_HANDOFF.md](V4_HANDOFF.md) "When Stuck"
3. [LIBRARIAN_REVIEW.md](LIBRARIAN_REVIEW.md) (documentation consistency, role definition)
4. Open an issue on GitHub

**Time to run benchmarks:** ~20 minutes (A100), ~2 hours (RTX 4050)  
**Time to understand the project:** ~1 hour (ONBOARDING + README + first benchmark)

Good luck! 🚀
