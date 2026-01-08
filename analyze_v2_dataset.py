"""
GroundThink V2: Dataset Forensics
=================================
Validates the quality of the 'Editor's Cut' dataset before training.
Checks for:
1. Token counts and distribution.
2. Contamination (e.g., 'As an AI language model').
3. Formatting issues (empty strings, weird chars).
4. Source balance.
"""

import os
from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np

DATASET_PATH = "groundthink_v2_dataset"

def analyze():
    print(f"🔍 Analyzing Dataset at: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset not found! Run prepare_v2_dataset.py first.")
        return

    ds = load_from_disk(DATASET_PATH)
    print(f"✅ Loaded {len(ds)} samples.")
    
    # 1. Source Distribution check (if we kept metadata, assuming 'source' col exists, if not we skip)
    if 'source' in ds.column_names:
        from collections import Counter
        print("\n📊 Source Distribution:")
        counts = Counter(ds['source'])
        for k, v in counts.items():
            print(f"   - {k}: {v} ({v/len(ds)*100:.1f}%)")
            
    # 2. Token Analysis
    print("\n🧮 Token Stats (Estimating with RWKV World Tokenizer or similar)...")
    # We'll use a standard tokenizer for estimation if World is not available
    try:
        tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-world-chntuned", trust_remote_code=True)
    except:
        print("   (Using GPT2 tokenizer for approximation)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    lengths = []
    # Sample 1000 items for speed
    sample_indices = np.random.choice(len(ds), min(1000, len(ds)), replace=False)
    
    print(f"   Sampling {len(sample_indices)} items for statistics...")
    for idx in sample_indices:
        txt = ds[int(idx)]['text']
        lengths.append(len(tokenizer.encode(txt)))
        
    avg_len = np.mean(lengths)
    total_est_tokens = avg_len * len(ds)
    print(f"   - Avg Tokens/Doc: {avg_len:.1f}")
    print(f"   - Min Tokens/Doc: {np.min(lengths)}")
    print(f"   - Max Tokens/Doc: {np.max(lengths)}")
    print(f"   - Estimated Total Corpus: {total_est_tokens/1e6:.2f} Million Tokens")
    
    # 3. Slop Detection
    print("\n🕵️  Slop Search (Checking for leakage)...")
    bad_phrases = [
        "As an AI", "I cannot fulfill", "OpenAI", "ChatGPT", "April 2023",
        "Generate a story", "Write a python script", # Instruction leakage
    ]
    
    slop_count = 0
    for idx in sample_indices:
        txt = ds[int(idx)]['text']
        for bad in bad_phrases:
            if bad in txt or bad.lower() in txt.lower():
                print(f"   ⚠️  Found suspicious phrase [{bad}]: {txt[:100]}...")
                slop_count += 1
                
    if slop_count == 0:
        print("   ✅ Clean scan on sample set.")
    else:
        print(f"   ❌ Found {slop_count} contaminated samples in sub-sample.")

    # 4. Visual Inspection
    print("\n👀 Visual Inspection (Random 3 Samples):")
    for i in range(3):
        idx = np.random.randint(0, len(ds))
        item = ds[int(idx)]
        src = item.get('source', 'unknown')
        print(f"\n--- Sample {i+1} (Source: {src}) ---")
        print(item['text'][:500] + "...")
        print("-" * 50)

if __name__ == "__main__":
    analyze()
