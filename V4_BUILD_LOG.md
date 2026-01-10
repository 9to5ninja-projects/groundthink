# GroundThink V4 Build Log

**Started:** 2026-01-09  
**Status:** In Progress  

---

## Build Session 1: 2026-01-09

### Created: `hybrid_v4.py`

**Spec (V4_DESIGN.md) vs Implementation:**

| Component | Spec | Implementation | Status |
|-----------|------|----------------|--------|
| Architecture | ParallelHybridBlock | ParallelHybridBlock | ✅ Match |
| RWKV-6 | `RWKV6Attention` from FLA | `RWKV6Attention` from FLA | ✅ Match |
| Mamba-2 | `Mamba2` from FLA | `Mamba2` from FLA | ✅ Match |
| Parallel execution | Both on same normalized input | Both on same normalized input | ✅ Match |
| Fusion | Learned gains (rwkv_gain, mamba_gain) | Learned gains init 0.5 | ✅ Match |
| FFN | 4x expansion, GELU | 4x expansion, GELU | ✅ Match |
| Normalization | RMSNorm | Custom RMSNorm (not nn.RMSNorm) | ⚠️ Minor deviation |
| Tied embeddings | Yes | Yes | ✅ Match |
| Default hidden | 128 | 128 | ✅ Match |
| Default layers | 8 | 8 | ✅ Match |

### Deviations from Spec

1. **RMSNorm**: Spec uses `nn.RMSNorm`, implementation uses custom `RMSNorm` class
   - Reason: Compatibility - `nn.RMSNorm` may not be available in all PyTorch versions
   - Impact: None - functionally identical

2. **RWKV6 num_heads**: Spec shows no explicit num_heads, implementation passes `num_heads=4`
   - Reason: Required for RWKV6 key_dim divisibility (key_dim=64 must divide by num_heads)
   - Impact: None - correct for hidden=128

3. **Mamba2 config**: Spec shows minimal config, implementation adds `expand`, `head_dim`, `num_heads`
   - Reason: Mamba2 requires `num_heads = expand * hidden_size / head_dim` per FLA config
   - Impact: None - required for correct operation

### API Discoveries (Not in Spec)

1. `RWKV6Attention.forward()` returns tuple `(output, attn_weights, past_kv)`, not tensor
2. `Mamba2.forward()` returns tensor directly
3. Mamba2 head formula: `num_heads = (expand * hidden_size) // head_dim`

### Tests Passed

- [x] Forward pass on CUDA (batch=2, seq=64, vocab=256)
- [x] Output shape correct: [2, 64, 256]
- [x] No dimension mismatches

### Runtime Warnings (Expected, Not Blocking)

```
1. "Detected Windows operating system. Triton does not have an official Windows release..."
   → FLA warns about Windows but works. Consider Linux for production.

2. "According to Bo, you are using a potentially buggy FLA implementation of RWKV..."
   → Hardcoded warning in FLA source, not suppressible. Cross-check with official RWKV-LM if publishing.

3. "The fast path is not available because `selective_state_update` is None..."
   → Needs mamba-ssm package, LINUX ONLY - uses C++ `and`/`or` keywords MSVC doesn't support.

4. "The CUDA backend is not available because `causal_conv1d` is None..."
   → Needs causal-conv1d package, LINUX ONLY - same MSVC incompatibility.
```

**Status:** Warnings 3 & 4 are **UNFIXABLE on Windows**. The CUDA kernel packages (mamba-ssm, causal-conv1d) 
use GCC/Clang-specific C++ syntax (`and`, `or` operators instead of `&&`, `||`) that MSVC doesn't support.

**Resolution:** FLA's Triton fallback is the **best available option on Windows**. For maximum performance,
use Linux where the native CUDA kernels can be installed.

**Performance Impact:** Triton fallback is ~2-3x slower than native CUDA kernels for Mamba2 operations.
For small-scale training (under 100M params), this is acceptable. For production training, use Linux.

### Environment Issues Fixed

1. **OpenMP conflict** - Added `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` before imports
2. **CUDA required** - FLA/Triton needs GPU tensors, model and inputs must be `.to(device)`

### Files Created

- `hybrid_v4.py` - Main model implementation
- `env_init.py` - Environment initialization helper

### Files Updated

- `V4_DESIGN.md` - Added runtime requirements and FLA API gotchas sections

---

## Build Session 2: 2026-01-09

### Task 1: Verify data_v030.py + tokenizer_v030.py

**Status:** ✅ VERIFIED WORKING

**Tests Run:**
1. `CharTokenizer` - encode/decode roundtrip: ✅ Match
2. `StatefulDataset` - instantiation: ✅ Works
3. Train/val split - separate indices: ✅ Verified (90/10 split)
4. Batch shapes correct: ✅ `[batch_size, seq_len]`
5. `get_val_batch()` - sequential val access: ✅ Works

**Key Details:**
- `StatefulDataset` splits data into parallel tracks (one per batch element)
- Train tokens are first 90%, val tokens are last 10% (no overlap)
- `is_new_doc` tensor returned for state reset logic
- `EOS_ID = 3` used as document separator

**No Changes Needed:** V3 data infrastructure is fully reusable for V4.

---

### Task 2: Create/Validate Test Dataset

**Status:** ✅ COMPLETE

**Dataset:** `shakespeare.txt` (Gutenberg)

| Metric | Value |
|--------|-------|
| File size | 1.06 MB |
| Total tokens | 1,115,392 |
| Vocab size | 97 (char-level) |
| Batch size | 4 |
| Seq length | 256 |
| Train steps | 980 |
| Val steps | 108 |
| Train/val split | 90/10 (no overlap) |

**Load Command:**
```python
from data_v030 import load_stateful_dataset
dataset, tokenizer = load_stateful_dataset(
    'shakespeare.txt', batch_size=4, seq_len=256, scale='8M', val_ratio=0.1
)
```

**Verification:**
- ✅ Train ends at token 250,963, val starts at 250,963 (no overlap)
- ✅ Batch shapes: `[4, 256]` for x and y
- ✅ `get_val_batch()` returns sequential validation batches

---

## Build Session 3: 2026-01-09

### Linux Environment Migration

**Status:** ✅ COMPLETE

**Reason:** Windows/WSL CUDA limitations. Native Linux required for:
- Full NVIDIA driver integration
- `causal-conv1d` and `mamba-ssm` CUDA kernels (won't compile on Windows)
- Filesystem performance during training

**Setup (see `setup_hybrid_env.sh` for full script):**

| Component | Version/Config |
|-----------|----------------|
| OS | Ubuntu (native Linux) |
| CUDA | 12.4 |
| PyTorch | 2.4.0+cu124 |
| GCC | 11.x (required for CUDA kernels) |
| causal-conv1d | v1.2.0 |
| mamba-ssm | v2.2.0 |
| GPU | RTX 4050 (sm_89) |

**Key fixes applied:**
- C++11 ABI flag: `-D_GLIBCXX_USE_CXX11_ABI=1`
- Architecture: `-gencode arch=compute_89,code=sm_89`
- Build with `--no-build-isolation`

---

### First Training Run

**Status:** ✅ COMPLETE

| Metric | Value |
|--------|-------|
| Model params | 3.8M |
| Throughput | ~33K tokens/sec |
| VRAM usage | ~422 MB |
| Final loss | 1.37-1.38 |
| Perplexity | ~3.0 |

**Gradient Health (needs attention):**
- RWKV/Mamba ratio: 0.15-0.16
- Status: WARN (outside [0.33, 3.0] range)
- Interpretation: RWKV gradients smaller than Mamba, possible imbalance

**Performance Validation:**
- Native CUDA kernels confirmed working
- 10-20x faster than Windows/Triton fallback

---

## Next Steps

- [ ] Investigate gradient ratio imbalance (RWKV too weak?)
- [ ] Consider adjusting `mamba_lr_mult` or RWKV initialization
