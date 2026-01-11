# GroundThink Documentation Map

**Updated:** 2026-01-11 (Post-Harmonization)  
**Status:** v5.0-Alpha | Phase 0.5 Planning

---

## Quick Navigation

| Category | Active Docs | Archive |
|----------|-------------|---------|
| **Strategy** | BASE_MODEL_CHARACTERIZATION.md (Phase 0), V0.5_ROADMAP.md | V4_STRATEGY_FULL.md |
| **Handoff** | V4_HANDOFF.md | - |
| **About** | ABOUT.md | - |
| **Observations** | OBSERVATION_SYNTHESIS.md | V4_FUSION_MODELS.md, V4_DIAGNOSTICS.md |
| **Reference** | V4_TESTING.md, V4_TRAINING_GUIDE.md, V4_DESIGN.md | V4_BLEND_RATIOS.md, V4_BUILD_LOG.md |
| **Validation** | STATEFUL_VALIDATION_GUIDE.md, CANARY_TESTS.md | - |
| **Research** | groundthink_architecture_research.md (3.4K lines) | - |

---

## Document Hierarchy

```
┌─ ACTIVE PLANNING (v5.0-Alpha)
│
├── BASE_MODEL_CHARACTERIZATION.md ⭐ PHASE 0 (CURRENT)
│   ├─ Pure RWKV-6 benchmark (Task 0.0.1)
│   ├─ Pure Mamba-2 benchmark (Task 0.0.2)
│   ├─ GPT-1 baseline (Task 0.0.3)
│   ├─ Comparative analysis (Task 0.0.4)
│   └─ Informs V0.5 fusion design
│
├── V0.5_ROADMAP.md ⭐ PHASE 1 (PENDING PHASE 0)
│   ├─ Section 1: Core Architecture (Tasks 0.1-0.6)
│   ├─ Acceptance criteria for all tasks
│   └─ Graduation criteria (V0.5 -> V1.0)
│
├── V4_HANDOFF.md
│   ├─ Current status (Phase 4.0 results)
│   ├─ Phase A: Documentation Cleanup (complete)
│   └─ Phase B: Implementation (next)
│
├── HARMONIZATION_REPORT.md
│   ├─ Executive Summary (GPT-2 parity, Mamba Paradox)
│   ├─ Research Gap Analysis (GRUs, Qualia, Sensors)
│   └─ Documentation Strategy
│
└── OBSERVATION_SYNTHESIS.md
    ├─ Executive Summary (18 observations)
    ├─ Key Inferences (attractor zone, imbalance)
    └─ Recommended Next Steps

┌─ RESEARCH BASELINE
│
└── groundthink_architecture_research.md (3.4K lines)
    ├─ Executive Summary
    ├─ Core Architecture Overview
    ├─ Pathway Specifications
    ├─ Hybrid Fusion Strategy
    └─ Critical Design Decisions (TBD)

┌─ V4 REFERENCE DOCS (Stable)
│
├── V4_STRATEGY.md (206 lines - trimmed)
│   ├─ Phase Results Summary
│   ├─ Critical Findings
│   ├─ Validation Test Results
│   └─ Key Lessons for V0.5
│
├── V4_TESTING.md
│   ├─ Gate Criteria (G1-G4)
│   ├─ State Tests (S0-S4)
│   └─ Diagnostic Tests (D1-D4)
│
├── V4_TRAINING_GUIDE.md
│   ├─ Training procedures
│   ├─ Hyperparameters
│   └─ Component warmup schedules
│
└── V4_DESIGN.md
    └─ ParallelHybridBlock architecture

┌─ VALIDATION FRAMEWORK
│
├── STATEFUL_VALIDATION_GUIDE.md
│   ├─ State Tracing Module
│   ├─ Grounding Score Calculator
│   └─ Diagnostic Test Implementations
│
├── CANARY_TESTS.md
│   ├─ C1-C6 test specifications
│   ├─ Test Suite Composition
│   └─ Canary Score Rubric
│
└── VALIDATION_ROADMAP.md
    └─ 3-week validation plan (Phase 3.9)

┌─ ARCHIVE (Historical Reference)
│
├── V4_STRATEGY_FULL.md (1084 lines - complete V4 backlog)
├── V4_FUSION_MODELS.md (Observations 1-18 detailed)
├── V4_DIAGNOSTICS.md (Diagnostic analysis)
├── V4_BLEND_RATIOS.md (Phase 3.7 experiments)
└── V4_BUILD_LOG.md (Build session logs)
```

---

## Reading Order by Role

### For New Contributors (Onboarding)
1. [ONBOARDING.md](ONBOARDING.md) - Start here
2. [V4_STRATEGY.md](V4_STRATEGY.md) - V4 achievements summary
3. [HARMONIZATION_REPORT.md](HARMONIZATION_REPORT.md) - Why V0.5?
4. [V0.5_ROADMAP.md](V0.5_ROADMAP.md) - What's next

### For Implementation (Phase B)
1. [V4_HANDOFF.md](V4_HANDOFF.md) - Current status
2. [V0.5_ROADMAP.md](V0.5_ROADMAP.md) - Task list with acceptance criteria
3. [V4_DESIGN.md](V4_DESIGN.md) - Architecture reference
4. [groundthink_architecture_research.md](groundthink_architecture_research.md) - Design decisions

### For Validation/Testing
1. [STATEFUL_VALIDATION_GUIDE.md](STATEFUL_VALIDATION_GUIDE.md) - Tool implementations
2. [CANARY_TESTS.md](CANARY_TESTS.md) - Test specifications
3. [V4_TESTING.md](V4_TESTING.md) - Gate procedures
4. [OBSERVATION_SYNTHESIS.md](OBSERVATION_SYNTHESIS.md) - Known issues

### For Strategic Planning
1. [HARMONIZATION_REPORT.md](HARMONIZATION_REPORT.md) - V4 assessment
2. [OBSERVATION_SYNTHESIS.md](OBSERVATION_SYNTHESIS.md) - 18 observations
3. [groundthink_architecture_research.md](groundthink_architecture_research.md) - V0.5 baseline
4. [SCALING_MILESTONES.md](SCALING_MILESTONES.md) - Long-term roadmap

---

## Key Changes (2026-01-11)

### Archived
- V4_STRATEGY.md → Trimmed to 206 lines (was 1084)
- V4_FUSION_MODELS.md → archive/ (39K)
- V4_DIAGNOSTICS.md → archive/ (21K)

### Created
- V0.5_ROADMAP.md - Tasks 0.1-0.6 with acceptance criteria
- HARMONIZATION_REPORT.md - Phase 4.0 post-mortem

### Updated
- OBSERVATION_SYNTHESIS.md - References now point to archive/
- V4_STRATEGY.md - References updated for archived docs
- V4_HANDOFF.md - Phase A complete, Phase B next

---

## File Size Reference

| File | Size | Status | Purpose |
|------|------|--------|---------|
| groundthink_architecture_research.md | 117K | Active | V0.5 research baseline |
| V4_TRAINING_GUIDE.md | 36K | Reference | Training procedures |
| STATEFUL_VALIDATION_GUIDE.md | 32K | Active | Validation tools |
| CANARY_TESTS.md | 25K | Active | Test specifications |
| V4_DESIGN.md | 21K | Reference | Architecture specs |
| HARMONIZATION_REPORT.md | 8K | Active | V4 post-mortem |
| V4_STRATEGY.md | 7.4K | Reference | V4 summary |
| V0.5_ROADMAP.md | 5K | Active | V0.5 task list |
| V4_HANDOFF.md | 5.4K | Active | Current status |
| OBSERVATION_SYNTHESIS.md | 7K | Active | Key findings |

---

**Governance:** Librarian Tier 2 (Navigation + Audit)  
**Next Review:** After Phase B implementation starts
