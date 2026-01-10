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

## Current Status

**Phase:** 1 - Foundation  
**Active Task:** NONE - Pick first available task from V4_STRATEGY.md  
**Blocking Issues:** None

**Task Selection Rule:** 
Pick the FIRST task in the backlog where:
1. Status is ⬜ PENDING (not ✅ COMPLETE or 🔄 IN PROGRESS)
2. All dependencies are ✅ COMPLETE

Tasks are ordered by priority. Do not skip ahead.

**Last Session (2026-01-09):**
- ✅ Task 3 COMPLETE - Built `hybrid_v4.py` - ParallelHybridBlock model verified working
- ✅ Created `V4_BUILD_LOG.md` - Tracks spec vs implementation
- ✅ Fixed environment issues (OpenMP, CUDA requirements)
- ✅ Updated `V4_DESIGN.md` with runtime requirements and FLA API gotchas

**Currently Available Tasks (dependencies met):**
- Task 1: Verify data_v030.py + tokenizer_v030.py work ← FIRST PRIORITY
- Task 4: Define Training Configuration
- Task 5: Pass Gate G1-G2

---

## Quick Context

**Goal:** Build RWKV-6 + Mamba-2 hybrid at 5M scale

**Critical files:**
- [V4_DESIGN.md](V4_DESIGN.md) - Architecture specification + runtime requirements
- [V4_STRATEGY.md](V4_STRATEGY.md) - Task backlog (ordered)
- [V4_BUILD_LOG.md](V4_BUILD_LOG.md) - What was actually built vs spec
- [V4_TESTING.md](V4_TESTING.md) - Testing framework

**Code files:**
- `hybrid_v4.py` - Model implementation (verified working)
- `env_init.py` - Environment setup helper

**FLA library locations:**
- RWKV-6: `fla/fla/layers/rwkv6.py` → `RWKV6Attention`
- Mamba-2: `fla/fla/layers/mamba2.py` → `Mamba2`

---

## What V3 Got Wrong (Don't Repeat)

V3 was scrapped because agents:
1. Built RWKV-7 instead of RWKV-6 (ignored spec)
2. Made up components instead of using FLA library
3. Didn't read documentation before coding

**V4 Rule:** Use FLA implementations. Do not substitute.

---

## Active Task Section

When you pick a task:
1. Write the task number and title here
2. Add your sub-task breakdown
3. Clear when complete

```
TASK: [number] [title]
Started: [date]
Agent: [session identifier if known]

Sub-tasks:
- [ ] ...
```

---

## Session Log

Record what you did:

```
[DATE] [TASK] - [OUTCOME]
---
2026-01-09 LINUX MIGRATION - Migrated to native Linux. CUDA kernels (causal-conv1d, mamba-ssm) now working. 33K tok/s achieved.
2026-01-09 FIRST TRAINING RUN - 3.8M model on Shakespeare. Loss 1.37, perplexity ~3.0. Gradient ratio WARN (0.15-0.16).
2026-01-09 Task 3 (Build Model) - Created hybrid_v4.py, verified forward pass on CUDA. See V4_BUILD_LOG.md for spec comparison.
2026-01-09 Environment fixes - Added env_init.py, documented OpenMP/CUDA requirements in V4_DESIGN.md
2026-01-08 Documentation - Fixed V4_STRATEGY.md, V4_DESIGN.md, created V4_HANDOFF.md
```

---

## FLA API Gotchas (MUST READ)

```python
# 1. RWKV6Attention returns a TUPLE
out_rwkv, _, _ = self.rwkv6(norm_x)  # NOT: out_rwkv = self.rwkv6(norm_x)

# 2. Mamba2 returns tensor directly
out_mamba = self.mamba2(norm_x)

# 3. Mamba2 head formula is MANDATORY
mamba_heads = (expand * hidden_size) // head_dim  # Must follow this exactly

# 4. ALL tensors MUST be on CUDA (Triton requirement)
model = model.to(device)
x = torch.randint(..., device=device)
```

---

## ✅ LINUX ENVIRONMENT (CURRENT)

**We have migrated to native Linux.** CUDA kernels now work.

| Package | Status |
|---------|--------|
| causal-conv1d v1.2.0 | ✅ Working |
| mamba-ssm v2.2.0 | ✅ Working |
| PyTorch 2.4.0+cu124 | ✅ Working |

**Performance:** ~33K tokens/sec (vs ~2.6K on Windows/Triton)

**Setup script:** `setup_hybrid_env.sh`

---

## ⚠️ WINDOWS LIMITATIONS (HISTORICAL)

**causal-conv1d and mamba-ssm CANNOT be installed on Windows:**
- These packages use GCC/Clang C++ syntax (`and`, `or` operators)
- MSVC doesn't support these - compile fails with "syntax error: missing ')' before identifier 'and'"
- **No workaround exists** - code would need upstream patches

**This is why we migrated to Linux.**

---

## When Stuck

**ASK THE USER.** Do not guess. Do not substitute.

---

*Next: Task 4 - Create train_v4.py. See V4_STRATEGY.md for details.*
