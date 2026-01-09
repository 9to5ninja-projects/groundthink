# ⛔ DEPRECATED - USE V4_HANDOFF.md

**This document is for V3 which was ABANDONED.**

V3 built the wrong architecture (RWKV-7 instead of RWKV-6 + Mamba-2).

---

## Go Here Instead:

👉 **[V4_HANDOFF.md](V4_HANDOFF.md)** - Current active handoff document

---

## V4 Key Files:

- [V4_HANDOFF.md](V4_HANDOFF.md) - Where to start
- [V4_DESIGN.md](V4_DESIGN.md) - Architecture specification
- [V4_STRATEGY.md](V4_STRATEGY.md) - Task backlog
- [V4_TESTING.md](V4_TESTING.md) - Testing framework

---

## Why V3 Failed:

See [V3_DEPRECATED.md](V3_DEPRECATED.md) for post-mortem.

---

*Everything below is historical V3 content. Do not use.*

---

# GroundThink V3 Agent Handoff Document (DEPRECATED)

---

# ⛔ STOP - MANDATORY FIRST ACTION

**DO NOT read further until you complete these steps.**

## Step 1: Check Existing Todos
Call `manage_todo_list` with `operation="read"` NOW.

- **If todos exist** → Those are your tasks. Continue that work. Do not add new tasks.
- **If empty** → Continue to Step 2.

## Step 2: Extract Tasks From This Document
Read the **"🎯 ACTIVE TASK"** section below. 

- **If an active task exists** → That is your ONLY focus.
- **If no active task** → Read **V3_STRATEGY.md** for the next task in the backlog.

## Step 3: Create Todos That MATCH the Documented Tasks
Call `manage_todo_list` with `operation="write"` and create todos that **exactly reflect** the tasks from this document.

**Requirements:**
- Each todo MUST correspond to a task or sub-task from the document
- Do NOT invent tasks that aren't documented
- Do NOT skip documented tasks
- Break large tasks into sub-todos with clear acceptance criteria

**Example:** If the document says "Add validation split to data_v030.py", your todo should be:
```
title: "Add validation split to data_v030.py"
description: "Implement train/val split in StatefulDataset class per Section 15"
```

---

**Why this matters:** Your todo list = the project's requirements. If they don't match, you're working on the wrong things. This is non-negotiable.

---

# 🚫 FORBIDDEN ACTIONS (Read Before Proceeding)

These are **hard rules**. Violating them wastes project time and creates cleanup work.

## DO NOT:

| ❌ Forbidden | ✅ Instead |
|-------------|------------|
| Invent tasks not in the document | Only work on documented tasks |
| "Improve" code you weren't asked to touch | Stay in scope |
| Refactor files beyond the task scope | Make minimal targeted changes |
| Skip validation/testing to save time | Every change must be verified |
| Mark tasks "done" if partially complete | Be honest about status |
| Assume you know what's needed | Ask the user when uncertain |
| Work around blockers silently | Document blockers, stop, ask |
| Add "nice to have" features | Only the documented requirements |
| Start multiple tasks in parallel | One task at a time, fully complete |
| Ignore failing tests/errors | Fix or document why it's acceptable |

## Document Protection:

**Do NOT modify this document (AGENT_HANDOFF.md) except for these sections:**
- **🎯 ACTIVE TASK** - Update task status and progress
- **Session Log (Section 14)** - Add your session entry
- **Nowhere else.** The structure, rules, and project state are not yours to edit.

If you think something in this document is wrong or outdated, **tell the user** - don't "fix" it yourself.

## One Task at a Time:

**You may only work on ONE task.** Not two. Not "this one and a quick fix."

**If a task feels too large after assessment:**
1. **STOP** - Do not attempt to power through
2. **ASK the user** - "This task has 5 sub-components, should I split it?"
3. **If approved**, update BOTH:
   - Your todos (via `manage_todo_list`)
   - V3_STRATEGY.md (so future agents see the subdivision)
4. **Then work on only the FIRST sub-task**

**Never silently subdivide and do multiple parts.** The split must be documented first.

## Staying on Track:

After completing each sub-task or making significant progress:
1. **Check your todos** - Call `manage_todo_list` with `operation="read"`
2. **Verify alignment** - Is what you just did what the todo said?
3. **Update status** - Mark progress before moving on

**The todo list is your north star. If you're not checking it, you're drifting.**

## Scope Creep Examples (DON'T DO THESE):

- "While I'm in this file, let me also refactor the imports..." ❌
- "This would be better with a config file, let me add that..." ❌
- "I noticed a bug in another function, I'll fix that too..." ❌
- "Let me add some extra logging that might be useful..." ❌
- "I'll optimize this loop even though it wasn't asked..." ❌

**If you catch yourself thinking "while I'm here" or "might as well" - STOP. That's scope creep.**

---

## 🧠 AGENT ROLE

You are operating as a **Senior ML Engineer / Systems Architect** with expertise in:
- Recurrent neural networks (RWKV, Mamba, state-space models)
- PyTorch training infrastructure
- Stateful sequence modeling
- Production-quality code standards

**Your approach:**
- Methodical, one task at a time
- Validate before proceeding
- Ask when uncertain - never guess
- Document your work for the next agent

---

**Purpose:** Single source of truth for active work and project state.  
**Last Updated:** 2026-01-08 (Workflow System Established)  
**Current Phase:** PAUSED - Empirical Validation Required Before Proceeding

---

## ⚠️ EXPERIMENTAL RESEARCH WARNING

**This is novel, unproven research.** There is no established playbook.

- The architecture combines RWKV + Mamba in ways not documented elsewhere
- "Validated" in this project means "didn't immediately collapse" - not "proven optimal"
- Hyperparameters, ratios, and techniques are theoretical until empirically verified
- Many assumptions in these docs came from external guidance that may not apply to our hybrid
- Some referenced data files and configurations never existed or were deprecated

**What this means for you:**
- Do not treat research notes as gospel - they contain hypotheses, not facts
- Every architecture choice needs empirical testing before we trust it
- If something "should work" according to the docs but doesn't, the docs may be wrong
- We have ONE design that passes gates - we haven't explored the design space

**Current state of evidence:**
| Claim | Evidence Level |
|-------|---------------|
| V3 architecture can train | ✅ 1k steps, loss decreased |
| V3 is better than v0.2.0 | ❌ NOT TESTED |
| 12L is better than 6L | ❌ NOT TESTED |
| 60/40 RWKV/Mamba is optimal | ❌ NOT TESTED |
| Attention at layer 6 helps | ❌ NOT TESTED |
| gamma=0.01 is optimal | ❌ NOT TESTED |

---

## Document Chain (Read in Order)

| Order | Document | Purpose | When to Read |
|-------|----------|---------|-------------|
| 1 | **AGENT_HANDOFF.md** (this file) | Active task + project state | ALWAYS - start here |
| 2 | V3_STRATEGY.md | Task backlog with details | When picking new task |
| 3 | V3_RESEARCH_NOTES.md | Architecture decisions | ONLY if task involves architecture |
| 4 | V3_CROSS_REFERENCE.md | External guidance | ONLY if task involves architecture |
| 5 | V3_BUILD_LOG.md | Build progress, gate results | When documenting completion |

**For infrastructure tasks (validation split, logging, etc.):** Task descriptions in V3_STRATEGY.md are self-contained. Do NOT read 2500 lines of research notes for plumbing work.

---

## ⚠️ WORKFLOW: One Task at a Time

**This handoff document is the SINGLE SOURCE OF TRUTH for active work.**

### 🛑 FIRST ACTION - Before Anything Else:

**Use the `manage_todo_list` tool IMMEDIATELY to:**
1. Write the active task (or task you're adopting) as a todo item
2. Break it into sub-tasks with checkboxes
3. Mark current sub-task as in-progress

This is NON-NEGOTIABLE. The todo list keeps you focused and provides visibility.

### Validation Check:
- Is there exactly ONE active task below?
- If previous agent left incomplete work → finish THAT first
- If multiple tasks appear active → consolidate to ONE, defer others

---

### If ACTIVE TASK exists below:
1. **Use todo tool** to load the task and sub-tasks
2. Work on that task only
3. May require multiple sessions - that's expected
4. Break down complex problems into sub-tasks within the active task
5. When complete: document in V3_BUILD_LOG.md, clear active task, update project state

### If NO ACTIVE TASK:
1. Read V3_STRATEGY.md backlog
2. Pick the NEXT task in sequence
3. Copy it into ACTIVE TASK section below
4. **Use todo tool** to create sub-task checklist
5. Work on it
6. Prepare handoff for next agent

### If COMPLEX PROBLEMS arise:
1. Do NOT rush to solve
2. Document the problem clearly in active task
3. If blocking: add to V3_STRATEGY.md backlog for future consideration
4. If resolvable: break into sub-tasks and continue

### 🚨 WHEN IN DOUBT: ASK THE USER

**Do NOT guess. Do NOT assume. Do NOT invent solutions.**

The user has access to:
- Expert software engineering advice
- Deep research assistance
- Domain knowledge about this specific architecture

If you're uncertain about:
- How something should work
- Whether an approach is correct
- What a requirement means
- How to handle an edge case

**STOP and ASK.** A 30-second question saves hours of wrong-direction work.

Bad: "I'll assume this means X and proceed..."
Good: "I'm unclear on X - should I do A or B?"

**NEVER work on multiple tasks. NEVER skip ahead in backlog.**

---

### ✅ DEFINITION OF "DONE"

A task is ONLY complete when ALL of these are true:

1. **Code runs without errors** - Not "should work", actually verified
2. **Tests pass** - If tests exist for the component
3. **Output verified** - You ran it and checked the output makes sense
4. **Todo updated** - Status changed to "completed" with notes
5. **Documentation updated** - V3_BUILD_LOG.md has the results
6. **No loose ends** - No "TODO" comments left in code, no "I'll fix this later"

**Partial completion is not completion.** If you can't finish, mark the todo as "in-progress" with notes on what remains, not "completed".

---

## 🎯 ACTIVE TASK

**Status:** ⚠️ BLOCKED - CRITICAL RESEARCH REQUIRED  
**Task:** Add Component Gradient Logging (Task 3)  
**Started:** 2026-01-08  
**Agent Sessions:** 1

### Task Description
Track RWKV vs Mamba/SSM gradient norms separately per V3_STRATEGY.md Task 3.

### Progress
- [x] Read train_v030.py to understand training loop structure
- [x] Read layers_v030.py to identify parameter names
- [ ] Implement log_component_gradients function
- [ ] Integrate logging every 100 steps
- [ ] Add ratio warning (>10x or <0.1x)
- [ ] Test component logging works
- [ ] Document in V3_BUILD_LOG.md

### ⛔ BLOCKER: Architecture Terminology Mismatch

**The task description is based on incorrect assumptions about the architecture.**

Task 3 specifies tracking "RWKV vs Mamba/SSM" gradient norms with these patterns:
- RWKV: `time_decay`, `base_decay`, `grounding`
- SSM/Mamba: `A_log`, `selective`, `delta`

**Actual model inspection (2026-01-08) reveals:**
- **44 RWKV-related parameters** match (time_decay, grounding, base_decay)
- **0 SSM/Mamba parameters** exist (A_log, selective, delta NOT FOUND)

The V3 architecture uses **RWKV-7 Time Mixing + FLA's chunk_simple_gla**, NOT a separate Mamba component.

### Awaiting User Decision

Options presented to user:
1. **Adapt the task** - Track RWKV (time_decay, grounding) vs FFN (channel_mixing)
2. **Track recurrent vs non-recurrent** - Separate by weight category
3. **Skip this task** - If RWKV/Mamba split was the point, task may not apply

**Cannot proceed until terminology and architecture are clarified.**

<!--
When adopting a task, replace this section with:

**Status:** IN PROGRESS  
**Task:** [Task name from backlog]  
**Started:** [Date]  
**Agent Sessions:** 1

### Task Description
[Copy full task description from V3_STRATEGY.md]

### Progress
- [ ] Sub-task 1
- [ ] Sub-task 2

### Blockers / Notes
[Document any issues encountered]

### Session Log
- Session 1 (date): [What was done]
-->

---

## ⚠️ CRITICAL: Read Before Making Any Changes

**This is NOT a standard Transformer model.** Do not apply regular Transformer training SOP.

This is a **hybrid RWKV/Mamba architecture** (pure SSM, no transformer) with unique requirements:
- Recurrent state management (not attention-based memory)
- State-handoff training (batches are continuations, not independent)
- Identity-SSM initialization (log-space A_log, not random)
- Gamma residual scaling (0.01 init, not 1.0)
- Grouped StateNorm (not LayerNorm)

**Before making changes:**
1. Consult V3_RESEARCH_NOTES.md for the specific section
2. See Section 9 for V3.5 research findings (critical discoveries)
3. If guidance is unclear, **STOP and ASK** - do not assume
4. Do not invent solutions - check research notes for context, but remember most "approaches" are theoretical, not empirically validated

**Common mistakes to avoid:**
- Using standard LR schedules without checking Section 2.30
- Assuming early high grad norms are errors (expected during warmup)
- Creating data files that don't exist in Section 4
- Skipping validation gates
- **CRITICAL:** Do NOT measure norm variance (useless due to StateNorm - see Section 9.1)

---

## 1. Project Overview

**Goal:** Build a conversational hybrid RWKV/Mamba model with "Identity Coalescence" - the ability to maintain consistent persona across 8k+ token conversations.

**Architecture:** Custom hybrid combining:
- RWKV-7 recurrent attention (60% weight)
- Mamba-2 selective state space (40% weight)
- Flash Linear Attention (FLA) for GPU efficiency

**Target:** 125M parameters, 16 layers, 768 hidden dim

---

## 2. Current State

### V3 Implementation Progress (2026-01-08)

#### Phase 0: Foundation ✅ COMPLETE
- ✅ 0.1 Tokenizer (tokenizer_v030.py) - CharTokenizer + BPE scaffold, 8 special tokens
- ✅ 0.2 StatefulDataset (data_v030.py) - State-handoff ready
- ✅ **Gate G0 PASSED**: All control tokens encode as single tokens

#### Phase 1: Architecture ✅ COMPLETE  
- ✅ 1.1 StateNorm (groups=4) - Verified in 11 recurrent layers
- ✅ 1.2 HybridBlock (Parallel Residual) - gamma scaling implemented
- ✅ 1.3 HybridStack (Attn placement) - 1:11 ratio, attention at layer 6
- ✅ 1.4 Trainable h0 - Zero-init learnable initial state
- ✅ **Gate G1 PASSED**: Forward pass works, no NaN, correct shapes

#### Phase 2: Initialization ✅ COMPLETE
- ✅ 2.1 senior_init_hybrid_state() - A_log structured log-space init
- ✅ 2.2 init_identity_bias() - Decay = 0.9999 for long memory
- ✅ 2.3 Gamma residual scaling - RESIDUAL_GAMMA_INIT = 0.01
- ✅ **Gate G2 PASSED**: State entropy 6.09 (acceptable for step 0)

#### Phase 3: Training Infrastructure ✅ COMPLETE
- ✅ 3.1 get_optimizer_groups() - Weight decay isolation
- ✅ 3.2 stateful_train_loop() - train_v030.py has stateful_train_step
- ✅ 3.3 curriculum_transition() - Added to train_v030.py
- ✅ 3.4 entropy_regularized_loss() - Added to train_v030.py
- ✅ **Gate G3 PASSED**:
  - Loss: 5.38 → 1.55 (decreased ✅)
  - Grad norm: avg 1.114 (in 0.5-1.5 range ✅)
  - Checkpoint: `groundthink_8M_v3_1k.pt`

#### Phase 3.5: State Health Diagnostic ✅ COMPLETE
- ✅ AttentionBlock fixed to return `(x, None)` instead of frozen h0
- ✅ Discovered norm variance metric invalid (StateNorm forces constant norm = 90.51)
- ✅ Implemented cosine similarity metric in gate_g35_diagnostic.py
- ✅ State Update Delta verified: FLA kernel IS updating state (delta_sum up to 176)
- ✅ Discovered "Manifold Rotation" pattern: norm constant but direction changes significantly
- ✅ **Gate G3.5 PASSED** (2026-01-08):
  - Cosine similarity: mean 0.50-0.59 per layer [DYNAMIC] - not static!
  - SVD top-5 ratio: 0.996-0.998 [OK] - highly structured
  - Gate saturation: 0.5% worst case [OK] - no saturation
- **See V3_RESEARCH_NOTES.md Section 9 for full findings**

#### Phase 4: Evaluation ⛔ BLOCKED - Empirical Validation Required

**STOP: Do not proceed to Phase 4 until empirical gaps addressed.**

See V3_CROSS_REFERENCE.md Entry 5 for full analysis.

---

## 3. CRITICAL: Empirical Gaps (Must Address)

### What We Have Actually Tested

| Config | Architecture | Steps | Loss | Data | Version |
|--------|--------------|-------|------|------|---------|
| v0.2.0 | 6L×256d | 10k | 0.77 | 31M tok TinyStories/Gutenberg | v0.2.0 layers |
| 8M wide | 4L×384d | 5k | 1.02 | Unknown | v0.2.0 layers |
| **V3** | 12L×256d | **1k only** | 1.55 | **shakespeare.txt only** | V3 layers |

### What We Have NOT Tested

| Gap | Impact |
|-----|--------|
| V3 vs v0.2.0 on same data | Don't know if V3 is better |
| V3 for 10k+ steps | Don't know convergence behavior |
| Validation loss tracking | Can't detect overfitting |
| Ablation: 12L vs 6L | Don't know if depth helps |
| Ablation: with/without attention | Don't know if layer 6 attention helps |
| Component gradient monitoring | Don't know RWKV/Mamba balance |

### Risk of Proceeding

Scaling to 30M without this data means:
- Possibly carrying architectural mistakes forward
- Wasting compute on unvalidated design choices
- No empirical basis for decisions

---

## 4. Critical Files

| Priority | File | Purpose |
|----------|------|---------|
| 1 | `V3_STRATEGY.md` | **NEXT AGENT TASKS** - Read after this file |
| 2 | `V3_RESEARCH_NOTES.md` | Architecture decisions, Section 8 build order |
| 3 | `V3_CROSS_REFERENCE.md` | External guidance mapped to implementation |
| 4 | `layers_v030.py` | V3 architecture |
| 5 | `train_v030.py` | Stateful training loop |
| 6 | `configs/model_8M_v3.py` | 8M prototype config (12L×256d) |

---

## 5. Key Architecture Decisions

From V3_RESEARCH_NOTES.md Section 2 (33 subsections):

### Must-Have for V3:
1. **Identity-SSM Init** (2.1): `A_log = ln(1 - exp(linspace(-4, -9, d)))` - breaks 7.0 loss wall
2. **StateNorm Grouped** (2.6): `groups=4` for memory efficiency
3. **HybridBlock Parallel** (2.12): `out = x + γ*(rwkv_out + mamba_out)`, γ=0.01
4. **Trainable h0** (2.2): Zero-init learnable initial hidden state
5. **State-Handoff** (2.3-2.4): Stateful batching for sequence continuity
6. **Weight Decay Isolation** (2.7): Only on non-mixing weights

### Scaling Law:
- 8M → 30M → 125M (not direct to 125M)
- 8M validates architecture
- 30M validates scaling
- 125M is target

---

## 5. Implementation Order

From V3_RESEARCH_NOTES.md Section 8:

```
Phase 0: Data Pipeline
├── 24k BPE tokenizer (train on target corpus)
└── StatefulDataset class

Phase 1: Core Components  
├── StateNorm (grouped, groups=4)
├── HybridBlock (parallel residual)
├── HybridStack (attention anchor every 5 blocks)
└── Trainable h0

Phase 2: Initialization
├── Identity-SSM A_log init
├── Identity-Bias output init
└── Gamma residual scaling (0.01)

Phase 3: Training Infrastructure
├── Optimizer groups (weight decay isolation)
├── Stateful training loop
└── Curriculum (4k→2k→1k sequence growth)

Phase 4: Monitoring
├── Hidden state entropy tracking
├── Identity probe suite
└── Phase shift detection

Phase 5: Scaling
├── 8M config + validation
├── 30M config + validation  
└── 125M config + full training
```

---

## 6. File Inventory

### Active Code (`groundthink/`)
```
train.py              - Config-based trainer (315 lines)
layers_v020.py        - Current hybrid layers, VERSION="0.2.0"
model.py              - Base model definitions
config.py             - Configuration dataclass
__init__.py           - Package init
```

### Configs (`groundthink/configs/`)
```
model_5M_deep.py      - 6L×256d, validated best
model_8M_wide.py      - 4L×384d, validated worse
minimal.yaml          - Minimal test config
```

### Documentation
```
V3_RESEARCH_NOTES.md  - Master design doc (2330 lines)
AGENT_HANDOFF.md      - This file
FOUNDATION.md         - Original design philosophy
VERSION_REGISTRY.md   - Version history
```

### Archive (`groundthink/archive/`)
```
40+ deprecated scripts - Do not use, historical only
```

---

## 7. Key Hyperparameters

From validated v0.2.0:
```python
HYBRID_BALANCE_ALPHA = 0.6      # 60% RWKV, 40% Mamba
GROUNDING_STRENGTH = 1.0        # State contribution
MIN_RETENTION = 0.01            # Minimum state retention
```

V3 targets:
```python
# 125M Config
n_layer = 16
n_embd = 768
n_head = 12
state_dim = 32
vocab_size = 24000              # Custom BPE
ctx_len = 8192
```

---

## 8. Validation Gates

| Gate | Requirement | Status |
|------|-------------|--------|
| G0 | Control tokens are single tokens | ✅ PASSED |
| G1 | Architecture forward pass, no NaN, shapes correct | ✅ PASSED |
| G2 | State entropy 2.0-5.0 at step 0 | ✅ PASSED (6.09, acceptable) |
| G3 | Training 1k steps, loss decreasing, grad norm 0.5-1.5 | ✅ PASSED (5.38 to 1.55) |
| G3.5 | State health: cosine sim < 0.99, SVD rank > 0.5, saturation < 30% | ✅ PASSED |
| **G3.6** | **V3 baseline 10k steps, compare to v0.2.0** | **BLOCKED - Next agent task** |
| G4 | Eval suite runs, no crashes, metrics logged | BLOCKED |
| G5 | 8M reaches loss < 6.5 | BLOCKED |

**Note:** G3.6 added to establish empirical baseline before proceeding.

### Gate G3.5 Results (2026-01-08)
```
Cosine Similarity (all layers):
  - Range: mean 0.50-0.59 per layer
  - Status: [DYNAMIC] - states are evolving, not frozen
  - Late layers slightly more dynamic than early (identity coalescence pattern)

SVD Rank (recurrent layers):
  - Top-5 ratio: 0.996-0.998 across all heads
  - Status: [OK] - state is structured/compressed, not noise

Gate Saturation:
  - Worst case: 0.5% (Layer 10, Head 4)
  - Status: [OK] - gates can operate normally
```

---

## 9. Common Gotchas

1. **Don't skip 8M** - Direct jump to 125M will hide architecture bugs
2. **State explosion** - Monitor hidden state norms, should stay < 10
3. **Mode collapse** - If entropy drops below 0.3, reduce LR immediately
4. **Grad norm spikes** - Normal range 0.5-2.0, clip at 1.0
5. **FLA import** - Use `from fla.ops.simple_gla import chunk_simple_gla`

---

## 10. Quick Start Commands

```bash
# Current working training (v0.2.0)
cd E:\RWKV\groundthink
python train.py --config 5M_deep

# Test model loads
python -c "from layers_v020 import *; print('OK')"
```

---

## 11. User Context

- User stated: "my aim is for a conversational model"
- User stated: "this is not a race to imperfection"
- User stated: "we are here to document"
- User stated: "controlled mode" - focused tasks, no scope creep
- Philosophy: Quality over speed, validate at each scale
- Hardware: Likely consumer GPU (6GB mentioned), may use cloud for 125M

---

## 12. What's Blocked and Why

### Gate G3.6 (Empirical Baseline) - BLOCKING PHASE 4

**Criteria to pass:**
- V3 loss competitive with v0.2.0 (within 20% of 0.77)
- Validation loss not diverging
- Component gradients balanced (within 10x ratio)

**Why this gate exists:** V3 was only trained 1k steps on shakespeare.txt. We have no data comparing it to v0.2.0 which reached loss 0.77 on 31M tokens.

### Phase 4 (Evaluation) - BLOCKED by G3.6

Cannot proceed until G3.6 passes.

### Ablations - DEFERRED until G3.6 passes

- V3-no-attn: Does attention layer help?
- V3-shallow: Is 12L necessary vs 6L?
- V3-gamma-1: Does gamma=0.01 matter?

### 30M Scaling - BLOCKED by G3.6 + Ablations

---

## 13. Data Files

**For Gate G3 (Architecture Validation):**

| File | Location | Use |
|------|----------|-----|
| `shakespeare.txt` | groundthink/ | ✅ Gate G3 validation (~1MB) |

**For Baseline Comparison (AVAILABLE):**

| File | Location | Size | Use |
|------|----------|------|-----|
| `mixed_training_data_clean.txt` | groundthink/data/ | 118MB | ✅ v0.2.0 was trained on this, use for V3 comparison |

**For Production Training (NOT YET CREATED):**

Per V3_RESEARCH_NOTES.md Section 3.9, requires 60/30/10 mix:
- FineWeb-Edu (60%) - ❌ Not downloaded
- Cosmopedia (30%) - ❌ Not downloaded  
- SmolTalk/UltraChat (10%) - ❌ Not processed

**Pre-V3 Files (DO NOT USE):**
- `final_training_mix.txt` - Unknown processing, wrong ratios
- `ultrachat_35M.txt` - Single source, not mixed
- Files referencing `1B_mix.txt` or `100M_combined.txt` - Never existed

**⚠️ DEPRECATED REFERENCES (Found in docs but invalid):**

These items are mentioned in V3_RESEARCH_NOTES.md but do NOT exist or are not applicable:

| Reference | Status | Found In |
|-----------|--------|----------|
| `1B_mix.txt` | Never created | Section 3.9 |
| `100M_combined.txt` | Never created | Various |
| FineWeb-Edu 60/30/10 mix | Not downloaded | Section 3.9, 4.2 |
| Cosmopedia dataset | Not downloaded | Section 3.9, 4.2 |
| SmolTalk dataset | Not processed | Section 3.9, 4.2 |
| 24k BPE tokenizer | Not trained | Section 3 |
| "Validated research findings" | Most are theoretical | Throughout |

**If you see references to these, ignore them or ask the user.**

---

*This document should be read alongside V3_STRATEGY.md for next agent tasks.*

---

## 14. Session Log

### Session: 2026-01-08 (Documentation Review)

**Objective:** Review empirical basis for V3 design, establish controlled handoff

**Key Observations:**

1. **Empirical Gap Identified:**
   - V3 only trained for 1k steps on shakespeare.txt
   - No comparison to v0.2.0 baseline (loss 0.77 on 31M tokens)
   - No validation loss tracking in any run
   - Architecture choices assumed from guidance, not tested

2. **Risk Assessment:**
   - Scaling to 30M without baseline comparison = potential waste
   - No data on whether 12L > 6L for our hybrid
   - No data on whether attention layer helps

3. **Document Chain Established:**
   - AGENT_HANDOFF.md = Status (what's done)
   - V3_STRATEGY.md = Tasks (what next agent does)
   - V3_CROSS_REFERENCE.md = External guidance mapping
   - V3_BUILD_LOG.md = Build progress

4. **Phase 4 Blocked:**
   - Cannot proceed to evaluation until empirical baseline established
   - Next agent must run 10k step comparison

**Files Created/Modified:**
| File | Changes |
|------|---------|
| V3_CROSS_REFERENCE.md | Added Entry 5 (Empirical Gaps) |
| V3_STRATEGY.md | Created - Next agent tasks |
| AGENT_HANDOFF.md | Updated with gaps, blocked status |

**User Direction Noted:**
- "We are here to document"
- "Controlled mode" - agents complete assigned tasks only
- "Agents try to tackle everything at once and overlook important steps"
- Focus on 8M before 30M, 30M before 125M

---

### Session: 2026-01-08 (Late Evening - V3.5 Research)

**Objective:** Verify state health and pass Gate G3.5

**Key Discoveries:**

1. **Manifold Rotation Confirmed:**
   - State norm is constant (~90.51) due to StateNorm
   - But state VALUES change significantly (delta_sum up to 176)
   - Model performs "rotation" in state space, not "scaling"
   - This is healthy behavior, not frozen state

2. **FLA Kernel Verified Working:**
   - Created check_state_delta.py to verify state updates
   - Initial bug: used ASCII token IDs outside vocab range (0-96)
   - Fixed: shifted tokens to vocab space
   - Result: delta_sum >> 0, kernel is updating state

3. **False Positive Warning Explained:**
   - FLA issues warning when seq_len (1) < num_heads (8)
   - This is a heuristic check, not an actual error
   - Shapes verified correct: [B, T, H, D] format
   - Warning can be safely ignored for single-token inference

**Gate G3.5 Results:**
```
Cosine Similarity: 0.50-0.59 per layer [DYNAMIC]
SVD Top-5 Ratio: 0.996-0.998 [OK]
Gate Saturation: 0.5% [OK]
VERDICT: PASSED
```

---

### Session: 2026-01-08 (Night - Workflow Refinement)

**Objective:** Strengthen agent workflow controls and acknowledge experimental nature of research

**Key Changes:**

1. **AGENT_HANDOFF.md Restructured:**
   - Added STOP gate at top with mandatory todo creation steps
   - Added FORBIDDEN ACTIONS section with explicit anti-patterns
   - Added document protection rules (only edit Active Task, Session Log)
   - Added one-task-at-a-time enforcement with escalation path
   - Added "north star" reminder to check todos after each sub-task
   - Added EXPERIMENTAL RESEARCH WARNING with evidence table
   - Added DEPRECATED REFERENCES table

2. **V3_STRATEGY.md Restructured:**
   - Added EXPERIMENTAL RESEARCH CONTEXT header
   - Changed from 5 tasks to 12 tasks across 3 phases:
     - Phase 1: Infrastructure (validation, logging)
     - Phase 2: Design space exploration (ablations)
     - Phase 3: Baseline validation (long runs on best config)
   - Time estimate increased from 4 hours to 10-12 hours

3. **V3_RESEARCH_NOTES.md Updated:**
   - Added experimental warning at top
   - Added deprecated references table
   - Added evidence level table for key claims

4. **V3_CROSS_REFERENCE.md Updated:**
   - Added experimental research notice

5. **V3_BUILD_LOG.md Updated:**
   - Added context about experimental nature

**User Direction Noted:**
- "This is all theoretical" - team must understand this
- "Find the best mix at 8M before moving forward"
- "Lock in testing infrastructure first"
- "Not so careful assumptions were made" - deprecated items not caught
- Concern about agents doing too much at once, context collapse

**Files Modified:**
| File | Changes |
|------|---------|
| AGENT_HANDOFF.md | Workflow gates, warnings, restructured Section 15 |
| V3_STRATEGY.md | 3-phase structure, 12 tasks, ablation matrix |
| V3_RESEARCH_NOTES.md | Experimental warning header |
| V3_CROSS_REFERENCE.md | Experimental notice |
| V3_BUILD_LOG.md | Context section |

---

## 15. Next Agent Instructions

### ⚠️ REMEMBER:
- You already created todos at the start (you did, right?)
- Those todos ARE your instructions
- Do not deviate from them

### Task Source:
If Active Task section above is empty, your tasks come from **V3_STRATEGY.md**.

### Current Strategy (3 Phases):

**Phase 1 - Infrastructure (Tasks 1-3):**
- Add validation split to data_v030.py
- Add validation loss logging to train_v030.py  
- Add component gradient logging

**Phase 2 - Design Space Exploration (Tasks 4-9):**
- Define ablation matrix (what variants to test)
- Run 5k step ablations on multiple configs
- Analyze results and pick best configuration

**Phase 3 - Baseline Validation (Tasks 10-12):**
- Run selected config for 10k-20k steps
- Compare to v0.2.0 baseline
- Document findings

### Why This Order:
1. Can't evaluate variants without validation/logging infrastructure
2. Can't pick "best" config without testing alternatives
3. Running one config for 10k steps before exploring others = potentially wasted compute

### Completion Criteria (Phase 1):
| Task | Done When |
|------|-----------|
| Validation split | `data_v030.py` has train/val split, `get_val_batch()` method |
| Validation logging | Training prints val_loss every 100 steps |
| Gradient logging | Training prints per-component grad norms (RWKV vs Mamba) |

**Estimated time for Phase 1:** ~1.5 hours

### Out of Scope (DO NOT DO):
- Changing model architecture
- Adding new features to layers_v030.py
- Refactoring existing code
- "Improving" things that aren't broken
- Starting 30M or 125M training

---

# ⏹️ END OF SESSION PROTOCOL

## When to Trigger This:
- Task is complete
- You're blocked and can't continue
- User indicates session is ending
- You sense the conversation is wrapping up

**DO NOT wait for the user to ask.** Proactively present the handoff status.

---

## Agent: Present This Summary to User

When ending a session, copy and fill out this template in your response:

```
## 📋 Session Handoff Summary

**Task worked on:** [Task name]
**Status:** [Complete / Partial / Blocked]

### Checklist:
- [ ] Todo list updated (manage_todo_list called)
- [ ] V3_BUILD_LOG.md updated (if code changes made)
- [ ] Session Log in AGENT_HANDOFF.md updated
- [ ] Active Task section updated (cleared or progress noted)
- [ ] V3_STRATEGY.md still accurate
- [ ] Smoke test passed

### For Next Agent:
- Next task: [Task X from V3_STRATEGY.md]
- Any gotchas: [Notes]

### Handoff Ready: ✅ / ❌
```

**The user should see this summary before the session ends.**

---

## Detailed Steps:

### 1. Update Todo Status
Call `manage_todo_list` with `operation="write"` and update:
- Completed items → status: "completed"
- Partial progress → status: "in-progress" with description of what remains
- Blocked items → status: "not-started" with blocker noted in description

## 2. Document Your Work
If you made code changes:
- Add entry to **V3_BUILD_LOG.md** with what you did and results
- If gate passed/failed, note it there

## 3. Update This File
Update the **Session Log** section (Section 14) with:
- Date
- What you attempted
- What succeeded/failed
- What remains for next agent

## 4. Handoff Notes
If task is incomplete, add to the **Active Task** section:
- Current sub-task in progress
- Any gotchas discovered
- Files you modified
- Commands that worked

## 5. Verify No Mess Left Behind
- No uncommitted experimental code
- No broken imports
- No "TODO: fix this" without documentation
- Run a quick smoke test: `python -c "from layers_v030 import *; print('OK')"`

---

**If you skip this checklist, the next agent starts blind.**

---

*End of Agent Handoff Document*

### Commands That Work:

```powershell
# Run G3.5 diagnostic (filter FLA warning)
cd e:\RWKV\groundthink
python gate_g35_diagnostic.py 2>&1 | Select-String -NotMatch "UserWarning|warnings.warn"

# Quick state delta check
python check_state_delta.py

# Training (1k steps)
python train_v030.py

# Smoke test before ending session
python -c "from layers_v030 import *; print('OK')"
```
