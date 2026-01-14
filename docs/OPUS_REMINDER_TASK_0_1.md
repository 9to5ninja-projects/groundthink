# OPUS CONTINUATION: GroundThink Task 0.1 → Next Steps

## CONTEXT
- **Repo:** github.com/9to5ninja-projects/groundthink
- **Version:** 0.5.1.1
- **Phase:** 1 (Twin Debate Architecture)

## COMPLETED
- Task 0.1 GRU Arbiter implemented (`ops/arbiter_gru.py`)
- Exploration notebook run (`notebooks/task_0_1_exploration.ipynb`)

## KEY FINDINGS FROM EXPLORATION

| Experiment | Result | Meaning |
|------------|--------|---------|
| 1: Constant Input | α = 0.499 ± 0.015 | ✓ No init bias |
| 2: Synthetic Divergence | α = 0.493 ± 0.020 | ⚠ Flat - no variance response untrained |
| 3-4: Real Outputs | α = 0.502 ± 0.0006 | ⚠ Flat - no response to 4x norm diff |
| 6: Hidden State | Norm 0.36 → 0.75 | ✓ GRU accumulates info |
| 7: Trainability | Loss -94% | ✓ IT LEARNS |

**Diagnosis:**
- Untrained GRU produces flat α ≈ 0.5 regardless of input
- Hidden state DOES evolve (norm doubles over sequence)
- Problem is `weight_proj` init (std=0.01 too small)
- Trainability confirmed: 94% loss reduction
- **GRU Arbiter architecture is VIABLE**

## DECISION POINT

GRU works but GLU/minGRU may be more efficient. User has added reference docs for both.

| Option | Pros | Cons |
|--------|------|------|
| Keep GRU | Proven works, implemented | More params, slower |
| minGRU | Simpler, fewer params | Need to implement |
| GLU | No recurrence, fastest | May lose temporal context |

**Need to decide before Task 0.2.**

## PHASE 0 FINDINGS (Reference)
- RWKV-6: amplifier (1.28x/layer, 5.5x total)
- Mamba-2: damper at layer (0.005x), amplifier as model (2.0x)
- BlinkDL init: mandatory for all

## FILES TO REVIEW
- `BASE_MODEL_CHARACTERIZATION.md`
- `V0.5_ROADMAP.md`
- `HANDOFF.md`
- `docs/minGRU.md` (new reference)
- `docs/GLU.md` (new reference)
- `docs/TASK_0_1_FINDINGS.md` (new)

## NEXT TASK OPTIONS
1. Re-spec arbiter with GLU/minGRU before continuing
2. Task 0.2: Residual Connections (if keeping GRU)

## USER WANTS
- Discuss architecture decision (GRU vs minGRU vs GLU)
- Use findings + Phase 0 data to inform design
- Proper tool usage for implementation work
