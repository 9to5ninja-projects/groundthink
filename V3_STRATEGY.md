# ⛔ DEPRECATED - SEE V3_DEPRECATED.md

**This document is for a build that used the wrong architecture.**

---

# V3 Strategy Document - Task Backlog (DEPRECATED)

**Created:** 2026-01-08  
**Purpose:** Ordered queue of tasks to be completed one at a time  
**Current Goal:** Lock in testing infrastructure, THEN explore design space at 8M

---

## ⚠️ EXPERIMENTAL RESEARCH CONTEXT

**This is novel research with no established playbook.**

- We have ONE architecture that passes gates - we have NOT explored the design space
- "Validated" means "didn't collapse" - not "proven optimal"
- Most hyperparameters (60/40 ratio, 12L depth, gamma=0.01) are theoretical choices
- External guidance may not apply to our specific hybrid architecture
- Some data files and configurations referenced in docs never existed

**Current evidence:**
- V3 can train (1k steps, loss 5.38 → 1.55 on shakespeare.txt)
- State is dynamic, not frozen (Gate G3.5 passed)
- That's it. Everything else is hypothesis.

**Strategy:**
1. **First:** Lock in testing/validation infrastructure (Tasks 1-3)
2. **Then:** Explore design space with proper metrics (Phase 2)
3. **Only then:** Pick best variant for 10k+ baseline (Phase 3)

---

## How This Document Works

**This is a BACKLOG, not a to-do list for one agent.**

1. Agent checks AGENT_HANDOFF.md for active task
2. If no active task: pick NEXT task from this backlog (in order)
3. Copy task to AGENT_HANDOFF.md "Active Task" section
4. **IMMEDIATELY use `manage_todo_list` tool** to create sub-task checklist
5. Work on that ONE task until complete
6. When done: clear from handoff, mark complete here, prepare for next agent

### Why the Todo Tool is Required

The `manage_todo_list` tool:
- Keeps agent focused on ONE task
- Provides user visibility into progress
- Prevents context drift and scope creep
- Creates natural checkpoints for multi-session work

**First action after reading handoff = write todo list. No exceptions.**

**Do NOT:**
- Work on multiple tasks at once
- Skip ahead in the backlog
- Proceed to Phase 4 or 30M scaling until backlog complete

### When Backlog is Complete

If all tasks are ✅ COMPLETE:
1. Update AGENT_HANDOFF.md project state
2. Check if Gate G3.6 criteria are met
3. If gate passes: add Phase 4 tasks to this backlog
4. If gate fails: add investigation/fix tasks to backlog
5. Document decision in V3_BUILD_LOG.md

### Adding New Tasks to Backlog

Only add tasks when:
- Current task reveals a blocker that needs separate work
- Gate criteria expose a gap
- User requests new scope

Format: Add to bottom of backlog table with next number, mark dependencies.

### When Stuck or Uncertain

**ASK THE USER. Do not guess.**

The user can get expert advice and deep research for any problem. A quick question is always better than:
- Guessing wrong and wasting time
- Making assumptions that break things
- Inventing solutions that don't fit the architecture

If blocked, document what you tried and what's unclear, then ask.

---

## Context (Why This Matters)

We have V3 architecture implemented and Gate G3.5 passed, but:
- Only 1k steps trained on shakespeare.txt
- No comparison to v0.2.0 baseline (which reached loss 0.77 on 31M tokens)
- No validation loss tracking
- No ablation studies to verify architecture choices
- **No exploration of design space** - we picked one configuration and ran with it

**Risk:** Scaling to 30M without this data could waste significant compute on unvalidated design.

**New Approach:** 
1. Lock in infrastructure (Tasks 1-3)
2. Run quick ablation studies at 5k steps to explore design space (Phase 2)
3. Pick best configuration based on evidence
4. Run that configuration for 10k+ steps (Phase 3)

---

## Task Backlog

### Phase 1: Infrastructure (Must Complete First)

| # | Task | Status | Depends On | Time |
|---|------|--------|------------|------|
| 1 | Add Validation Split to Data Pipeline | ✅ COMPLETE | - | 30 min |
| 2 | Add Validation Loss Logging | ✅ COMPLETE | Task 1 | 30 min |
| 3 | Add Component Gradient Logging | ⬜ PENDING | - | 30 min |

**Gate:** Infrastructure complete when all three tasks done and verified working.

### Phase 2: Design Space Exploration (After Infrastructure)

| # | Task | Status | Depends On | Time |
|---|------|--------|------------|------|
| 4 | Define Ablation Matrix | ⬜ PENDING | Tasks 1-3 | 30 min |
| 5 | Run Ablation: V3 Baseline (5k) | ⬜ PENDING | Task 4 | 1 hr |
| 6 | Run Ablation: Shallow 6L (5k) | ⬜ PENDING | Task 4 | 1 hr |
| 7 | Run Ablation: No Attention (5k) | ⬜ PENDING | Task 4 | 1 hr |
| 8 | Run Ablation: 50/50 Ratio (5k) | ⬜ PENDING | Task 4 | 1 hr |
| 9 | Analyze Ablation Results | ⬜ PENDING | Tasks 5-8 | 1 hr |

**Gate:** Pick best configuration based on evidence, not assumption.

### Phase 3: Baseline Validation (After Best Config Selected)

| # | Task | Status | Depends On | Time |
|---|------|--------|------------|------|
| 10 | Run Selected Config (10k-20k steps) | ⬜ PENDING | Task 9 | 2-4 hrs |
| 11 | Compare to v0.2.0 Baseline | ⬜ PENDING | Task 10 | 30 min |
| 12 | Document Findings & Update Docs | ⬜ PENDING | Task 11 | 1 hr |

**Legend:** ⬜ PENDING | 🔄 IN HANDOFF | ✅ COMPLETE

---

## Phase 1 Task Details

### Task 1: Add Validation Split to Data Pipeline

**Status:** ⬜ PENDING

**File:** `data_v030.py`  
**Time:** ~30 minutes  
**Scope:** Modify StatefulDataset to support train/val split

**Requirements:**
- Add `val_ratio` parameter (default 0.1)
- Create separate validation tracks
- Ensure validation data is NOT in training tracks
- Add `get_val_batch()` method

**Acceptance Criteria:**
- [ ] Running `python -c "from data_v030 import *; print('OK')"` works
- [ ] Validation batches do not overlap with training batches
- [ ] Document changes in V3_BUILD_LOG.md

**Example Todo List for This Task:**
```
1. Read data_v030.py to understand StatefulDataset structure [in-progress]
2. Add val_ratio parameter to __init__
3. Modify track creation to split train/val
4. Implement get_val_batch() method
5. Test import works
6. Test no overlap between train/val
7. Document in V3_BUILD_LOG.md
```

---

### Task 2: Add Validation Loss Logging

**Status:** ⬜ PENDING  
**File:** `train_v030.py`  
**Time:** ~30 minutes  
**Scope:** Add validation loss calculation every N steps

**Requirements:**
- Calculate validation loss every 100 steps
- Log format: `Step {step} | Train Loss: {train} | Val Loss: {val}`
- Store validation losses in checkpoint

**Acceptance Criteria:**
- [ ] Validation loss logged during training
- [ ] Losses saved in checkpoint dict
- [ ] No significant slowdown (val eval should be fast)
- [ ] Document changes in V3_BUILD_LOG.md

---

### Task 3: Add Component Gradient Logging

**Status:** ⚠️ BLOCKED - CRITICAL RESEARCH REQUIRED  
**File:** `train_v030.py`  
**Time:** ~30 minutes  
**Scope:** Track RWKV vs Mamba gradient norms separately

### ⛔ BLOCKER (2026-01-08)

**Architecture terminology mismatch discovered:**
- Task assumes RWKV + Mamba hybrid with separate components
- Actual V3 uses RWKV-7 Time Mixing + FLA (no Mamba module)
- SSM/Mamba patterns (`A_log`, `selective`, `delta`) do NOT exist in model
- 44 RWKV params found, 0 Mamba params found

**Awaiting user decision on how to proceed.**

---

**Original Requirements (may be outdated):**
- Categorize parameters by component (RWKV/Mamba/Other)
- Log ratio every 100 steps
- Warn if ratio exceeds 10x

**Parameter names in layers_v030.py (CORRECTED):**
- RWKV/Recurrent components: `time_decay`, `base_decay`, `grounding`
- SSM/Mamba components: ~~`A_log`, `selective`, `delta`~~ **NOT PRESENT**
- Other: everything else

**Implementation:**
```python
def log_component_gradients(model):
    rwkv_norms = []
    ssm_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            if any(x in name for x in ['time_decay', 'base_decay', 'grounding']):
                rwkv_norms.append(norm)
            elif any(x in name for x in ['A_log', 'selective', 'delta']):
                ssm_norms.append(norm)
    
    rwkv_avg = sum(rwkv_norms) / len(rwkv_norms) if rwkv_norms else 0
    ssm_avg = sum(ssm_norms) / len(ssm_norms) if ssm_norms else 0
    ratio = rwkv_avg / (ssm_avg + 1e-9)
    
    return rwkv_avg, ssm_avg, ratio
```

**Acceptance Criteria:**
- [ ] Component grad norms logged
- [ ] Ratio warning implemented (warn if >10x or <0.1x)
- [ ] Document changes in V3_BUILD_LOG.md

---

## Phase 2 Task Details

### Task 4: Define Ablation Matrix

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Define what we're testing and why

**Purpose:** Before running experiments, document what we're testing so results are interpretable.

**Ablation Matrix (Proposed):**

| Variant | Layers | Width | RWKV/Mamba | Attention | Gamma | Purpose |
|---------|--------|-------|------------|-----------|-------|---------|
| V3-baseline | 12 | 256 | 60/40 | Yes (L6) | 0.01 | Current default |
| V3-shallow | 6 | 256 | 60/40 | Yes (L3) | 0.01 | Is depth necessary? |
| V3-no-attn | 12 | 256 | 60/40 | No | 0.01 | Does attention help? |
| V3-balanced | 12 | 256 | 50/50 | Yes (L6) | 0.01 | Is 60/40 optimal? |

**Note:** This matrix is a PROPOSAL. Discuss with user before running if uncertain.

**Acceptance Criteria:**
- [ ] Matrix reviewed and approved by user (or confirmed as above)
- [ ] Each variant has clear hypothesis
- [ ] Config files or CLI flags defined for each variant

---

### Task 5: Run Ablation: V3 Baseline (5k)

**Status:** ⬜ PENDING  
**Time:** ~1 hour runtime  
**Scope:** Establish baseline metrics for comparison

**Requirements:**
- Run V3 default config for 5k steps
- Use `mixed_training_data_clean.txt` (or shakespeare.txt if unavailable)
- Log: train loss, val loss, component gradients, final loss
- Save checkpoint

**Metrics to record:**
```
| Variant | Steps | Final Train Loss | Final Val Loss | RWKV Grad | Mamba Grad | Ratio |
```

---

### Task 6: Run Ablation: Shallow 6L (5k)

**Status:** ⬜ PENDING  
**Time:** ~1 hour runtime  
**Scope:** Test if 12 layers is necessary

**Hypothesis:** v0.2.0 used 6L and reached 0.77. Maybe 6L is sufficient for 8M scale.

**Config changes:**
- `n_layer = 6` (from 12)
- Attention at layer 3 (middle)
- Everything else same

---

### Task 7: Run Ablation: No Attention (5k)

**Status:** ⬜ PENDING  
**Time:** ~1 hour runtime  
**Scope:** Test if attention layer helps

**Hypothesis:** The attention anchor every N blocks is theorized to help. Does it actually?

**Config changes:**
- Remove attention layer entirely
- All layers are HybridBlocks
- Everything else same

---

### Task 8: Run Ablation: 50/50 Ratio (5k)

**Status:** ⬜ PENDING  
**Time:** ~1 hour runtime  
**Scope:** Test if 60/40 RWKV/Mamba is optimal

**Hypothesis:** 60/40 was chosen based on external guidance. Maybe 50/50 works better for our hybrid.

**Config changes:**
- `HYBRID_BALANCE_ALPHA = 0.5` (from 0.6)
- Everything else same

---

### Task 9: Analyze Ablation Results

**Status:** ⬜ PENDING  
**Time:** ~1 hour  
**Scope:** Compare all variants and pick best configuration

**Create comparison table:**

| Variant | Final Loss | Val Loss | Grad Ratio | Notes |
|---------|-----------|----------|------------|-------|
| V3-baseline | ? | ? | ? | |
| V3-shallow | ? | ? | ? | |
| V3-no-attn | ? | ? | ? | |
| V3-balanced | ? | ? | ? | |

**Decision criteria:**
- Lower loss is better
- Val loss should track train loss (no overfitting)
- Grad ratio should be 0.1x - 10x (balanced training)

**Output:**
- Selected best variant with justification
- Document any surprising findings
- If multiple variants are close, may need longer runs

---

## Phase 3 Task Details

### Task 10: Run Selected Config (10k-20k steps)

**Status:** ⬜ PENDING  
**Time:** ~2-4 hours runtime  
**Scope:** Longer run on best configuration from Phase 2

**Requirements:**
- Use configuration selected in Task 9
- Run 10k steps minimum, 20k if time allows
- Full logging (train loss, val loss, component grads)
- Save checkpoint

---

### Task 11: Compare to v0.2.0 Baseline

**Status:** ⬜ PENDING  
**Time:** ~30 minutes  
**Scope:** Fair comparison to previous best

**v0.2.0 reference:**
- 5.5M params (6L × 256d)
- Loss 0.77 at 10k steps on 31M tokens
- No validation tracking
- layers_v020.py architecture

**Comparison table:**

| Metric | v0.2.0 | Selected V3 | Winner |
|--------|--------|-------------|--------|
| Params | 5.5M | ~8M | - |
| Train Loss @ 10k | 0.77 | ? | ? |
| Val Loss @ 10k | N/A | ? | - |
| Training Stability | ? | ? | ? |

**Possible outcomes:**
- V3 clearly better → proceed to 30M scaling
- V3 worse → investigate, may need to reconsider architecture
- V3 similar → need longer runs or different data to differentiate

---

### Task 12: Document Findings & Update Docs

**Status:** ⬜ PENDING  
**Time:** ~1 hour  
**Scope:** Comprehensive documentation update

**Update these files:**
1. **V3_BUILD_LOG.md** - Full ablation results, selected config, rationale
2. **AGENT_HANDOFF.md** - Update current state, evidence table
3. **V3_STRATEGY.md** - Write Phase 4 tasks OR investigation tasks
4. **V3_RESEARCH_NOTES.md** - Add empirical findings section (if significant)

**Key documentation:**
- What we tested
- What we learned
- What we're now confident about (with evidence)
- What remains uncertain

---

## What NOT To Do

- Do NOT proceed to 30M scaling until Phase 3 complete
- Do NOT modify architecture during ablations (test what exists)
- Do NOT skip the analysis step (Task 9)
- Do NOT run all ablations in parallel without infrastructure verified first
- Do NOT assume any configuration is "right" - let the data decide

---

## Files You Will Modify

### Phase 1:
| File | Changes |
|------|---------|
| data_v030.py | Add validation split |
| train_v030.py | Add val loss + component logging |

### Phase 2-3:
| File | Changes |
|------|---------|
| configs/ | May need variant config files |
| V3_BUILD_LOG.md | Ablation results |
| AGENT_HANDOFF.md | Updated evidence table |
| V3_STRATEGY.md | Next phase tasks |

---

## Time Budget

| Phase | Tasks | Estimate |
|-------|-------|----------|
| Phase 1 | Infrastructure (1-3) | 1.5 hours |
| Phase 2 | Ablations (4-9) | 5-6 hours |
| Phase 3 | Baseline + Docs (10-12) | 3-4 hours |
| **Total** | | ~10-12 hours |

**Note:** This can be split across multiple agent sessions. Each task is designed to be completable in one session.

---

## Questions? 

If blocked:
1. **ASK THE USER FIRST** - they have expert access
2. Check V3_RESEARCH_NOTES.md ONLY for architectural questions
3. Do NOT read research notes for simple infrastructure tasks - the task description here is sufficient

**Remember:** This is experimental research. If results don't match expectations, that's DATA, not failure. Document everything.

*End of Strategy Document*
