# GroundThink Scientific Registry
# ==============================================================================
# This registry tracks major experimental versions, dataset mixtures, architecture
# specifications, and hypothesis results. All changes must be logged here.
# ==============================================================================

PROJECT_NAME = "GroundThink"
CURRENT_VERSION = "V4.2-Alpha (Phase 2 Complete) — See CHANGELOG.md"
DATE_INITIATED = "2026-01-07"
LAST_UPDATED = "2026-01-09"

# ==============================================================================
# VERSION HISTORY
# ==============================================================================

## VERSION 1.0: "The Proving Ground"
- **Status:** COMPLETED (Jan 7, 2026)
- **Goal:** Verify Hybrid Architecture stability on A100. Prove "Label Leakage" fix.
- **Model:** 
    - Param: 1B
    - Layers: 24
    - Dim: 2048
    - Type: SelectiveWKV (RWKV Gates -> Mamba Kernel)
- **Dataset (V1):**
    - Method: Streaming Interleaved
    - Total Tokens: ~2B (Partial epoch)
    - Mix:
        - 80% FineWeb-Edu (Sample-10BT)
        - 20% TinyStories
- **Result:**
    - Identity Mapping bug fixed (Labels shifted).
    - Loss converged 11.2 -> 5.5 in 500 steps.
    - Output: Coherent babble ("humantohists").
    - "Gold Run" validated infrastructure.

## VERSION 2.0: "The Scientific Baseline" (CURRENT)
- **Status:** IN PREPARATION
- **Goal:** Establish scaling laws and memory density independent of parameter count.
- **Hypothesis:** "High-density data mix can force memory emergence in small (125M) model."
- **Model:**
    - Param: 125M (Standard Small)
    - Layers: 12
    - Dim: 768
    - Type: SelectiveWKV_V2 (Split Optimizer Groups + Triton)
- **Dataset (V2) - "The Editor's Cut":**
    - Method: Pre-downloaded, Curated, deterministic shuffle.
    - Total Samples: 200,000
    - Est. Tokens: ~200M - 400M (High epoch repetition allowed)
    - **Composition:**
        1. **Logic (40%)**: `FineWeb-Edu` (HuggingFaceFW/fineweb-edu)
           - *Filter:* Score >= 4 (Highly educational), Length > 1000 chars.
           - *Role:* Reasoning, structure, academic tone.
        2. **Memory (30%)**: `PG19` (deepmind/pg19)
           - *Filter:* Length > 8000 chars (Force >2k token context).
           - *Role:* Long-range dependency, narrative arc, state saturation.
        3. **Chat (20%)**: `OpenHermes-2.5` (teknium/OpenHermes-2.5)
           - *Format:* User/Assistant turns.
           - *Role:* Instruction following, conversational flow.
        4. **Grounding (10%)**: `TinyStories` (roneneldan/TinyStories)
           - *Filter:* Length > 200 chars.
           - *Role:* Grammar fundamentals, simple object permanence.
- **Infrastructure:**
    - Script: `train_groundthink_v2_125m.py`
    - Prep: `prepare_v2_dataset.py`

# ==============================================================================
# DATASET METADATA (V2 Detailed Analysis)
# ==============================================================================
# To be filled after running prepare_v2_dataset.py on A100
# - Actual Token Count: [PENDING]
# - Size on Disk: [PENDING]
# - Vocabulary Coverage: [PENDING]
