# Data Directory

**Last Updated:** 2026-01-10

---

## Current Datasets

### wikitext103/
**Source:** HuggingFace `wikitext/wikitext-103-raw-v1`  
**Downloaded:** 2026-01-10  
**Purpose:** Standard language modeling benchmark for V5 GPT-2 comparison

| File | Lines | Purpose |
|------|-------|---------|
| train.txt | 2,330,058 | Training data (~100M tokens) |
| valid.txt | ~4,000 | Validation set |
| test.txt | ~4,300 | Held-out test set |

**Why WikiText-103:**
- Industry standard for LM benchmarks (used in GPT-2 papers)
- Large enough for meaningful training (~100M tokens)
- Clean Wikipedia text, well-documented
- Allows direct comparison with published results

**Usage:**
```python
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
```

---

### shakespeare.txt
**Source:** Andrej Karpathy's char-rnn  
**Purpose:** Sanity testing, quick overfit tests, char-level experiments

**NOT for:** Final benchmarks, production training, GPT-2 comparison

---

## Archived (Untrusted)

Moved to `archive/old_data/`:
- fineweb_5m.txt — Unknown provenance, deprecated
- fineweb_10k.txt — Unknown provenance, deprecated

---

## Tokenization

All V5 benchmarks use **BPE tokenization** (not char-level).

| Tokenizer | Vocab | File | Purpose |
|-----------|-------|------|---------|
| WikiText BPE | 16,000 | tokenizer_wikitext.json | V5 benchmarks |

**Rule:** Same tokenizer for both GPT-2 and GF-MH to ensure fair comparison.

---

## Adding New Data

When adding new datasets:
1. Document source URL
2. Record download date
3. State purpose clearly
4. Note any preprocessing done
5. Update this README

**Do not use undocumented data files.**
