# GroundThink V2 - Project Status & Organization

**Last Updated**: January 8, 2026

## 🎯 Project Goal
Train a conversational hybrid RWKV/Mamba model at small scale, then scale up.

---

## 📊 Training Results

### Experiment Log (Jan 8, 2026)

| Model | Config | Steps | Loss | Speed | Notes |
|-------|--------|-------|------|-------|-------|
| v0.2.0 | 6L×256d (5.5M) | 10k | 0.77 | 52k tok/s | Old data mix |
| v0.2.0 | 6L×256d (5.5M) | 5k | 0.97 | 53k tok/s | New 39M mix |
| v0.2.0 | 4L×384d (8.3M) | 5k | 1.02 | 41k tok/s | Wide config, stable |
| **v0.2.0** | **6L×256d (5.5M)** | **50k** | **0.71** | **54k tok/s** | **Best so far** |

### 50k Run Details (Jan 8, 2026)
- **Time**: 63 minutes
- **Warmup**: 2000 steps
- **States**: Stable at 90.5 throughout
- **Best loss**: 0.7097

**Generation Quality:**
- Narrative: Coherent sentences ("little girl named Lily", "loved to sing")
- Short prompts: Garbage (needs context to warm up)
- Conversation: Emerging! "Can you help me? I want to see him..."

**Key Findings:**
1. Loss plateaued ~0.72 around step 36k
2. TinyStories dominates output (70% of training data)
3. Conversation responses mixed with narrative patterns
4. Short prompts fail - model needs context window

---

## 📁 Directory Structure

```
E:/RWKV/
├── data/                          # Training data (ACTIVE)
│   ├── final_training_mix.txt     # 149.5 MB - Ready for training!
│   ├── all_dialogue_data.txt      # 55.5 MB - All dialogue sources combined
│   ├── narrative_data_clean.txt   # 118 MB - TinyStories + Gutenberg
│   ├── dialogue_training_data.txt # 19 MB - TV dialogues only
│   └── cache/                     # Downloaded raw datasets
│
├── groundthink/                   # Core model code
│   ├── layers.py                  # Original layers (v0.1.0)
│   ├── layers_v020.py             # Fixed hybrid balance (v0.2.0) ✓
│   ├── model.py                   # Model architecture
│   ├── train_v020.py              # Training script with versioning
│   ├── train_hybrid_v2.py         # Alternative training script
│   └── data/                      # Old data location (legacy)
│
├── models/                        # Pre-trained RWKV models (reference)
│   ├── RWKV-x070-World-0.1B-*.pth
│   ├── RWKV-x070-World-0.4B-*.pth
│   └── rwkv7-g1c-2.9b-*.pth
│
├── fla/                           # Flash Linear Attention library
│
└── scripts/                       # Data preparation (to organize)
    ├── prepare_all_dialogue_data.py
    └── create_training_mix.py
```

---

## 🧠 Model Versions

### v0.1.0 - Initial Hybrid (DEPRECATED)
- **File**: `groundthink_small.pt`, `groundthink_mixed.pt`
- **Issue**: Broken hybrid balance formula
- **Status**: ❌ Do not use

### v0.2.0 - Fixed Hybrid Balance (CURRENT)
- **File**: `groundthink_v020_10k_5M.pt`
- **Architecture**: 6 layers, dim=256, 8 heads, head_dim=32
- **Parameters**: 5.5M
- **Training**: 10k steps on mixed data
- **Loss**: 3.38 → 0.77
- **Speed**: 52k tok/s locally
- **Balance Formula**: `w_combined = alpha * w_base + (1 - alpha) * w_selective`
- **Alpha**: 0.6 (60% base, 40% selective)
- **Status**: ✓ Working, ready for more training

---

## 📊 Training Data

### Current Mix (final_training_mix.txt)
| Component | Samples | Tokens (est.) | Ratio |
|-----------|---------|---------------|-------|
| Narrative (TinyStories+Gutenberg) | 199,606 | ~31M | 70% |
| Dialogue (sampled) | 85,545 | ~8M | 30% |
| **TOTAL** | **285,151** | **~39M** | 100% |

### Dialogue Sources (150k total available)
| Dataset | Dialogues | Description |
|---------|-----------|-------------|
| Prosocial-Dialog | 120,038 | Multi-turn ethical conversations |
| PersonaChat | 17,878 | Persona-based casual chat |
| DailyDialog | 10,396 | Daily life conversations |
| TV Dialogue | 2,287 | Screenplay dialogues |

### Narrative Sources
- TinyStories GPT-4 (~2.7M stories)
- Project Gutenberg classics (47 books)

---

## ❓ Key Decision: Fresh Training vs Continue?

### Option A: Continue from v0.2.0 checkpoint
**Pros:**
- Already at loss 0.77, good starting point
- Faster convergence
- Model has learned basic patterns

**Cons:**
- Trained on different data mix (old mix had less dialogue variety)
- May have learned suboptimal patterns
- Distribution shift from new data

### Option B: Fresh training on new data
**Pros:**
- Clean slate with better data mix
- No legacy patterns to unlearn
- 30% dialogue ratio from start

**Cons:**
- Starts from scratch
- Takes longer to converge

### 🎯 RECOMMENDATION: **Fresh Training (Option B)**
Reasons:
1. New data mix is significantly different (150k unique dialogues vs 2.3k repeated)
2. 5.5M model trains fast (~3 min for 10k steps)
3. Want to validate the architecture with proper data
4. Can always fine-tune later

---

## 🚀 Next Steps

### Immediate (Today)
1. [x] Organize project files
2. [ ] Decide fresh vs continue (see above)
3. [ ] Start training run with new data mix
4. [ ] Monitor loss curves

### Short Term
1. [ ] Train to 50k steps
2. [ ] Evaluate conversation quality
3. [ ] Benchmark against similar SSM models
4. [ ] Document hyperparameter findings

### Scale Up (After Validation)
1. [ ] Scale to 25M params
2. [ ] Scale to 125M params
3. [ ] Consider A100/cloud training

---

## 🔧 Key Files Reference

### Training
- `groundthink/train_v020.py` - Main training script with versioning
- `groundthink/layers_v020.py` - Fixed hybrid layers

### Data Preparation
- `prepare_all_dialogue_data.py` - Downloads all dialogue datasets
- `create_training_mix.py` - Creates final 70/30 mix

### Model
- `groundthink/model.py` - GroundThinkLM architecture
- `groundthink/config.py` - Model configurations

---

## 📝 Training Command (Proposed)

```bash
cd E:/RWKV/groundthink
python train_v020.py --data ../data/final_training_mix.txt --fresh
```

---

## ⚠️ Files to Clean Up (Later)
- `groundthink/deepseek_python_*.py` - AI-generated drafts
- `groundthink/train_*.py` (except train_v020.py) - Old training scripts
- `groundthink/data/` - Move to main data/ folder
