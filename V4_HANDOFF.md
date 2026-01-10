# V4 Agent Handoff Document

**Purpose:** Single source of truth for agent continuity  
**Updated:** 2026-01-09  
**Repository:** https://github.com/9to5ninja-projects/groundthink

---

## ⚠️ MANDATORY WORKFLOW

**Before doing ANYTHING else:**
1. Read this document completely
2. Use `manage_todo_list` tool to write your task breakdown
3. Only then begin work

**This is not optional. The user enforces this.**

---

## 🐧 ENVIRONMENT: NATIVE LINUX

**We are working in a native Linux (Ubuntu) environment via VS Code Remote-SSH.**

**CRITICAL:**
- Use Linux syntax for ALL commands (`/home/m_tes/groundthink`, not Windows paths)
- Terminal commands use bash shell syntax
- File operations use forward slashes `/`
- Python virtual environment is at `.venv/` (already in .gitignore)
- Git operations work normally

**Example paths:**
- ✅ Correct: `/home/m_tes/groundthink/train_v4.py`
- ❌ Wrong: `C:\Users\...\train_v4.py` or `\home\m_tes\...`

---

## 📝 EDITING BEST PRACTICES

**CRITICAL: Make small, incremental edits to prevent errors, truncation, and timeouts.**

**When creating or editing files:**
- ❌ **DO NOT** try to write 500+ line documents in one operation
- ❌ **DO NOT** batch 10+ file edits together
- ✅ **DO** break large documents into logical sections (50-150 lines each)
- ✅ **DO** edit files one at a time
- ✅ **DO** verify each edit before moving to the next

**Why this matters:**
- Large operations frequently timeout or get truncated
- Errors in large batches are hard to diagnose
- Incremental edits are easier to review and rollback
- User can provide feedback between steps

**Example workflow:**
1. Create document outline/structure first
2. Fill in section 1, commit
3. Fill in section 2, commit
4. Continue until complete

---

## 🚦 VALIDATION GATES (QUALITY CHECKPOINTS)

**All V4+ development must pass validation gates before proceeding.**

**Gate documentation:** See [V4_TESTING.md](V4_TESTING.md#validation-gates-quality-gates-for-all-v4-development) for full procedures.

**Quick reference:**
- **G1** (Forward pass): Run after building any model - ensures no NaN/shape errors
- **G2** (Init entropy): Run before first training - checks initialization (2.0-5.0 = healthy)
- **G3** (Train 1K steps): Run after 1000 steps - loss should decrease, grad norm 0.5-1.5
- **G3.5** (State health): Run before extended training - checks state evolution (cosine <0.99)
- **G4** (Component balance): Run when analyzing results - gradient ratio should be 0.3-3.0

**When to check gates:**
- After model build → G1
- Before training → G1 + G2
- After 1K steps → G3
- Before extended training → G3.5 + G4
- Before scaling up → Full suite G1-G4

**Current status:** Task 5 passed G1-G2, need to verify G3 status for 5000 step run.

---

## Current Status

**Phase:** Phase 1 Tasks 8-12 COMPLETE → Task 13 (Extended Training) NEXT  
**Last Updated:** 2026-01-09 (Build Session 10 continued)  
**Status:** All optimizations applied, gradient imbalance FIXED. Ready for extended training.

**Completed This Session:**
- ✅ Batch+AMP optimization: 186K tok/s (+586% vs baseline)
- ✅ Gradient imbalance FIXED: mamba_lr_mult 2.0→0.5
- ✅ train_v4.py updated with optimal config + AMP support
- ✅ V4_STRATEGY.md updated (Tasks 8-12 complete)

**Optimized Metrics (Final):**

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Throughput | 27,140 tok/s | 186,398 tok/s | **+586%** |
| Peak VRAM | 46.9 MiB | 184.9 MiB | Still < 200 MiB |
| Gradient Ratio | 0.15-0.16 ❌ | 0.7-1.3 ✅ | **FIXED** |
| Loss (200 steps) | 9.17→3.22 | 9.17→2.51 | Faster convergence |

**Config Changes Applied to train_v4.py:**
- `batch_size`: 8 → 64
- `seq_len`: 256 → 64 (for faster iteration)
- `use_amp`: False → True
- `mamba_lr_mult`: 2.0 → 0.5 (fixes G4 gradient imbalance)

See [V4_BUILD_LOG.md - Session 10](V4_BUILD_LOG.md#build-session-10-2026-01-09) for details.

---

## Next Agent Instructions

**Current Priority:** Task 13 - Extended Training Run (L complexity, ~4-8 hours)

**Objective:** Run 5000+ steps with optimal config to validate convergence.

**What to do:**
1. Run `python train_v4.py` (config already optimized)
2. Monitor loss curve, gradient ratio, throughput
3. Verify G1-G4 gates pass throughout
4. Save checkpoint at completion
5. Document results in V4_BUILD_LOG.md Session 11

**Expected metrics:**
- Throughput: ~186K tok/s
- Gradient ratio: 0.3-3.0 throughout
- Loss: Decreasing and stabilizing

**Before starting:**
1. Read this document completely ✓
2. Use `manage_todo_list` tool to write task breakdown (REQUIRED)
3. Run training: `python train_v4.py`

---

## Completed Tasks Summary

**Phase 0:** ✅ COMPLETE (Session 7-8)
- All CUDA kernels integrated and validated (G0-G4 pass)

**Task 7:** ✅ COMPLETE (Session 9)
- benchmark_suite.py created
- Baseline: 21K tok/s, 46.9 MiB, stability PASS
- V4.5_FUSION_VARIANTS.md created (kernel fusion research)
- data_loader.py + tokenizer.py standardized

**Tasks 8-12:** ✅ COMPLETE (Session 10)
- Task 8: Batch=64 + AMP = 186K tok/s (+586%)
- Task 9: Combination tests completed
- Task 10: Optimal config selected
- Task 11: Training dynamics analyzed
- Task 12: Gradient imbalance FIXED (mamba_lr_mult 2.0→0.5)

**Key Fix:** Gradient ratio was 0.15 (RWKV weaker). Changed mamba_lr_mult from 2.0 to 0.5. Now stable at 0.7-1.3.

---

## Implementation History

**Compiler fix applied:** CXX=/usr/bin/g++ CC=/usr/bin/gcc (absolute paths)
**RWKV-6 CUDA:** Compiles on first use, caches for subsequent imports
**Fallback mechanism:** Uses prototype if CUDA fails
- Test coverage: 7 individual tests + 5 gate validations

**Build Log:** See [V4_BUILD_LOG.md - Session 10](V4_BUILD_LOG.md#build-session-10-2026-01-09) for optimization details

---

## Quick Context

**Goal:** Build RWKV-6 + Mamba-2 hybrid at 5M scale

**Critical files:**
- [V4_DESIGN.md](V4_DESIGN.md) - Architecture specification + runtime requirements
- [V4_STRATEGY.md](V4_STRATEGY.md) - Task backlog (ordered)
- [V4_BUILD_LOG.md](V4_BUILD_LOG.md) - What was actually built vs spec
- [V4_TESTING.md](V4_TESTING.md) - Testing framework
- [V4.5_OPTIMIZATION.md](V4.5_OPTIMIZATION.md) - Performance optimization & monitoring guide

**Code files:**
- `hybrid_v4.py` - Hybrid model (5M params, working)
- `train_v4.py` - Training script with monitoring
- `benchmark_suite.py` - Reusable B1/B2/B3 benchmarks
- `data_loader.py` - Data pipeline (from archive)
- `tokenizer.py` - Tokenization (from archive)
- `fla_replacements.py` - Component bridge (CUDA first, fallback)
- `rwkv6_prototype.py` - PyTorch RWKV-6 reference
- `rwkv6_cuda_wrapper.py` - CUDA wrapper with JIT
- `test_phase0_complete.py` - Gate validation suite

**New documentation:**
- `V4.5_FUSION_VARIANTS.md` - Kernel fusion research & benchmarks

**Component Status:**
- RWKV-6: CUDA wrapper → PyTorch prototype (fallback)
- Mamba-2: mamba-ssm native CUDA kernels

---

## What V3 Got Wrong (Don't Repeat)

V3 was scrapped because agents:
1. Built RWKV-7 instead of RWKV-6 (ignored spec)
2. Made up components instead of using real implementations
3. Didn't read documentation before coding

**V4 Rule:** Use verified implementations (mamba-ssm, our RWKV-6 prototype). Do not substitute.

---

## Session Log

**Most recent first:**

```
2026-01-09 TASKS 9-12 COMPLETE - Gradient imbalance FIXED (mamba_lr_mult 2.0→0.5, ratio 0.7-1.3). train_v4.py optimized. 186K tok/s. Task 13 NEXT.
2026-01-09 TASK 8 COMPLETE - Quick win optimizations: 6.1x throughput (27K→166K tok/s) via batch=64. AMP +19% at batch=32. torch.compile blocked by PyTorch bug.
2026-01-09 TASK 7 COMPLETE - benchmark_suite.py created, baseline recorded: 21K tok/s, 46.9 MiB. Task 8 NEXT.
2026-01-09 MODULE REORGANIZATION - data_loader.py, tokenizer.py moved from archive. V4.5_FUSION_VARIANTS.md created.
2026-01-09 STRATEGY UPDATED - Task 7 (Baseline Profiling) sub-tasks defined, Task 8 (Quick Wins) linked to Task 7 benchmarks.
2026-01-09 COMPREHENSIVE PHASE 0 TESTING - Integrated CUDA wrapper, added 7 tests + 5 gates. All gates running.
2026-01-09 CUDA WRAPPER INTEGRATED - fla_replacements.py now tries CUDA wrapper first, falls back to prototype.
2026-01-09 COMPILER FIX - rwkv6_cuda_wrapper.py now sets CXX=/usr/bin/g++ CC=/usr/bin/gcc (absolute paths).
2026-01-09 PHASE 0 COMPLETE - Created fla_replacements.py, hybrid_v4.py forward pass works. All CUDA kernel tasks done.
2026-01-09 RWKV-6 CUDA WRAPPER - Created rwkv6_cuda_wrapper.py, compiles with CXX=g++ env fix.
2026-01-09 RWKV-6 PROTOTYPE - Created rwkv6_prototype.py, G1 gate passed (CPU + GPU).
2026-01-09 MAMBA-2 VERIFIED - mamba-ssm already installed with working CUDA kernels.
```

---

## ✅ LINUX ENVIRONMENT (CURRENT)

**Native Linux with full CUDA kernel support.**

| Package | Version | Status |
|---------|---------|--------|
| PyTorch | 2.4.0+cu124 | ✅ Working |
| mamba-ssm | 2.2.6 | ✅ Working (CUDA kernels) |
| causal-conv1d | (via mamba-ssm) | ✅ Working |
| CUDA Toolkit | 12.4 | ✅ Working |
| GCC | 11.5.0 | ✅ Working |
| nvtop | 3.0.2 | ✅ Installed |

**Compiler Settings:**
- CXX: `/usr/bin/g++` (required for torch extensions)
- CC: `/usr/bin/gcc`
- Set automatically in rwkv6_cuda_wrapper.py

**CUDA Configuration:**
- GPU: RTX 4050 (sm_89)
- Compute capability: 8.9
- Registers available: ~65K per thread
- Max shared memory: 96 KB per block

---

## When Stuck

**ASK THE USER.** Do not guess. Do not substitute.

See [V4_STRATEGY.md](V4_STRATEGY.md) for ordered task backlog.
