# Requirements Analysis

## Actual Dependencies (What We Use)

### Core (Required)

| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `torch` | ≥2.1.0 | Model, training, CUDA | Core framework |
| `mamba-ssm` | ≥2.2.0 | Mamba2 implementation | Brings triton, einops |
| `causal-conv1d` | ≥1.5.0 | Mamba2 dependency | CUDA kernel |
| `tokenizers` | ≥0.15.0 | BPE tokenization | HuggingFace Rust lib |
| `einops` | ≥0.7.0 | Tensor operations | Used in models |
| `ninja` | any | CUDA kernel compilation | JIT build |
| `packaging` | any | Version parsing | mamba-ssm dep |
| `pyyaml` | any | Config files | train_v4.py |

### CUDA Kernel (Our Custom Build)

| Component | Source | Purpose |
|-----------|--------|---------|
| `RWKV-CUDA/wkv6` | Submodule | RWKV6 attention kernel |
| `rwkv6_cuda_wrapper.py` | Local | JIT compile + wrapper |
| `rwkv6_prototype.py` | Local | PyTorch fallback |

**Note:** We built our own RWKV6 CUDA kernel wrapper. FLA library is NOT used.

### Optional (Not Currently Used)

| Package | In requirements.txt? | Actually Used? | Action |
|---------|---------------------|----------------|--------|
| `transformers` | ✅ | ❌ | Remove (mamba-ssm brings it) |
| `datasets` | ✅ | ❌ | Remove |
| `bitsandbytes` | ✅ | ❌ | Remove |
| `accelerate` | ✅ | ❌ | Remove |
| `scipy` | ✅ | ❌ | Remove |
| `tensorboard` | ✅ | ❌ | Remove |
| `psutil` | ✅ | ❌ | Remove (only in archive/) |

---

## Minimum Hardware Requirements

### GPU (Required)

| Scale | Min VRAM | Recommended | Notes |
|-------|----------|-------------|-------|
| 3.5M (Tiny) | 4 GB | 6 GB | RTX 3060, 4050 |
| 8M (Small) | 6 GB | 8 GB | RTX 3070, 4060 |
| 30M (Medium) | 12 GB | 16 GB | RTX 3080, 4080 |
| 125M (Large) | 24 GB | 40 GB | RTX 3090, A6000 |

**CUDA Compute Capability:** ≥7.0 (Volta or newer)

### CPU

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Cores | 4 | 8+ |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50 GB (for checkpoints) |

### Software

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| CUDA | 12.1+ |
| Linux | Ubuntu 22.04+ (native Linux required) |
| GCC | 11+ (for kernel compilation) |

**Windows:** Not supported for training (CUDA kernel JIT fails). WSL2 works.

---

## Cleaned requirements.txt

```
# Core
torch>=2.1.0
mamba-ssm>=2.2.0
causal-conv1d>=1.5.0
tokenizers>=0.15.0
einops>=0.7.0

# Build tools
ninja
packaging
wheel

# Config
pyyaml
```

---

## What We Built vs What We Use

### Our Custom Code

| File | Purpose | Replaces |
|------|---------|----------|
| `cuda_backends.py` | RWKV6 + Mamba2 imports | FLA library |
| `rwkv6_cuda_wrapper.py` | CUDA kernel wrapper | FLA RWKV6 |
| `rwkv6_prototype.py` | PyTorch fallback | None |
| `RWKV-CUDA/` | CUDA kernel source | Official RWKV kernels |

### Third-Party We Actually Use

| Package | What For |
|---------|----------|
| `mamba-ssm` | Mamba2 implementation (unchanged) |
| `causal-conv1d` | Mamba2 convolution (unchanged) |
| `tokenizers` | BPE tokenization |
| `torch` | Everything |

### Third-Party We Removed Dependency On

| Package | Was Used For | Status |
|---------|--------------|--------|
| FLA (`flash-linear-attention`) | RWKV6 + Mamba2 | **REMOVED** — built our own |
| `transformers` | Tokenizers? | **UNUSED** — we use `tokenizers` directly |
| `datasets` | Data loading? | **UNUSED** — we have custom loaders |
