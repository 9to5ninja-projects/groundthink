# ⛔ V3 DEPRECATED - DO NOT USE

**Date:** 2026-01-08  
**Reason:** Architecture mismatch - built RWKV-7 instead of requested RWKV-6 + Mamba-2

---

## What Happened

The user explicitly requested **RWKV-6 + Mamba-2 hybrid** architecture.

Agents repeatedly built **RWKV-7 + Grounding** instead, ignoring instructions.

This was discovered when Task 3 (component gradient logging) failed because the expected Mamba parameters (`A_log`, `selective`, `delta`) don't exist - because Mamba was never implemented.

---

## V3 Files - ALL DEPRECATED

| File | Status | Notes |
|------|--------|-------|
| layers_v030.py | ❌ DEPRECATED | Built RWKV-7, not RWKV-6+Mamba-2 |
| train_v030.py | ❌ DEPRECATED | Training loop may be salvageable but untrusted |
| data_v030.py | ⚠️ POSSIBLY OK | Data pipeline is architecture-agnostic |
| tokenizer_v030.py | ⚠️ POSSIBLY OK | Tokenizer is architecture-agnostic |
| V3_RESEARCH_NOTES.md | ❌ DEPRECATED | Mixed correct goals with wrong implementation |
| V3_STRATEGY.md | ❌ DEPRECATED | Tasks designed for wrong architecture |
| V3_CROSS_REFERENCE.md | ❌ DEPRECATED | References wrong architecture |
| V3_BUILD_LOG.md | ❌ DEPRECATED | Documents wrong build |
| gate_g35_diagnostic.py | ❌ DEPRECATED | Validates wrong architecture |
| check_state_delta.py | ❌ DEPRECATED | Tests wrong architecture |

---

## What V3 Actually Built (For The Record)

- **TimeMixing**: RWKV-7 style with FLA kernel acceleration
- **GroundingMechanism**: Custom stabilizer with conv + base_decay
- **ChannelMixing**: Standard FFN
- **HybridBlock**: Container with gamma residual scaling
- **AttentionBlock**: Simple attention anchor
- **StateNorm**: Grouped RMS normalization

**What was missing:**
- StableStateSSM (Mamba component)
- A/B/C/D matrices
- Selective scan
- Separate SSM optimizer groups
- Actual RWKV-6 formulation

---

## Do Not Reference These Documents for V4

- V3_RESEARCH_NOTES.md - contaminated with RWKV-7 assumptions
- V3_STRATEGY.md - tasks for wrong architecture
- V3_CROSS_REFERENCE.md - external guidance mapped to wrong implementation
- FOUNDATION.md - may contain correct goals but agents ignored it

---

## V4 Starts Fresh

V4 must implement what was actually requested:
- **RWKV-6** (not RWKV-7)
- **Mamba-2** (actual SSM with A/B/C/D)
- **True hybrid** with separate components and gradient tracking

---

*This document exists to prevent future agents from using V3 code or docs.*
