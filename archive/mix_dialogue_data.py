"""
Mix Dialogue Data with Existing Training Data
==============================================
Combines:
- Existing: TinyStories + Gutenberg classics (narrative)
- New: TV/Movie dialogue (natural conversation)

Target mix: 70% existing + 30% dialogue
"""

import os
import random
from pathlib import Path
from tqdm import tqdm
import hashlib

# ==========================================
# CONFIGURATION  
# ==========================================
DATA_DIR = Path("data")
EXISTING_FILE = DATA_DIR / "mixed_training_data_clean.txt"
DIALOGUE_FILE = DATA_DIR / "dialogue_training_data.txt"
OUTPUT_FILE = DATA_DIR / "combined_training_data.txt"

# Mix ratio (adjust as needed)
# With 2.3k dialogue samples and 200k narrative, realistic max is ~5% dialogue
# Unless we repeat heavily (RWKV-LM says 3-4x is fine)
MAX_DIALOGUE_REPEAT = 4  # Per RWKV-LM recommendation
DIALOGUE_RATIO = 0.30  # Target 30% dialogue (will adjust if not enough data)

SEED = 42

# ==========================================
# MAIN
# ==========================================
def load_lines(filepath):
    """Load lines from file, filtering short ones"""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if len(line.strip()) > 50]
    return lines


def deduplicate(lines):
    """Remove duplicate lines by hash"""
    seen = set()
    unique = []
    for line in lines:
        sig = line[:200] + line[-200:]
        h = hashlib.md5(sig.encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(line)
    return unique


def mix_and_save():
    print("=" * 60)
    print("Mixing Dialogue + Narrative Data")
    print("=" * 60)
    
    # Load existing data
    print(f"\n📖 Loading existing narrative data: {EXISTING_FILE}")
    existing = load_lines(EXISTING_FILE)
    
    if not existing:
        # Try alternate filename
        alt_file = DATA_DIR / "mixed_training_data.txt"
        print(f"   Trying alternate: {alt_file}")
        existing = load_lines(alt_file)
    
    if not existing:
        print("❌ No existing training data found!")
        return
    
    print(f"   Loaded: {len(existing):,} lines")
    
    # Load dialogue data
    print(f"\n💬 Loading dialogue data: {DIALOGUE_FILE}")
    dialogue = load_lines(DIALOGUE_FILE)
    
    if not dialogue:
        print("❌ No dialogue data found! Run prepare_dialogue_data.py first")
        return
    
    print(f"   Loaded: {len(dialogue):,} lines")
    
    # Calculate target mix
    # With dialogue ratio, we want dialogue to be DIALOGUE_RATIO of final dataset
    # If dialogue is smaller, we'll repeat it (per RWKV-LM recommendation: repeat 3-4x)
    
    existing_count = len(existing)
    dialogue_count = len(dialogue)
    
    # Calculate how many dialogues we need for the target ratio
    # If existing = 200k, dialogue_ratio = 30%, we want final 70% existing, 30% dialogue
    # So dialogue_needed = existing_count * DIALOGUE_RATIO / (1 - DIALOGUE_RATIO)
    dialogue_needed = int(existing_count * DIALOGUE_RATIO / (1 - DIALOGUE_RATIO))
    
    print(f"\n📊 Target Mix (ratio: {DIALOGUE_RATIO:.0%} dialogue):")
    print(f"   Existing narrative: {existing_count:,}")
    print(f"   Dialogue available: {dialogue_count:,}")
    print(f"   Dialogue needed for ratio: {dialogue_needed:,}")
    
    # Repeat dialogue data if needed
    if dialogue_count < dialogue_needed:
        repeat_times = (dialogue_needed // dialogue_count) + 1
        print(f"   Repeating dialogue {repeat_times}x (RWKV-LM: repeat 3-4x is good)")
        dialogue = dialogue * repeat_times
    
    # Now sample to exact target
    dialogue = random.sample(dialogue, min(dialogue_needed, len(dialogue)))
    
    print(f"\n   Final mix:")
    print(f"   - Narrative: {existing_count:,} ({100*(1-DIALOGUE_RATIO):.0f}%)")
    print(f"   - Dialogue: {len(dialogue):,} ({100*DIALOGUE_RATIO:.0f}%)")
    print(f"   - Total: {existing_count + len(dialogue):,}")
    
    # Combine and shuffle
    print("\n🔀 Combining and shuffling...")
    combined = existing + dialogue
    random.shuffle(combined)
    
    # Deduplicate
    combined = deduplicate(combined)
    
    print(f"   After dedup: {len(combined):,} lines")
    
    # Calculate stats
    total_chars = sum(len(s) for s in combined)
    est_tokens = total_chars / 4
    
    print(f"\n📈 Final Statistics:")
    print(f"   Total lines: {len(combined):,}")
    print(f"   Total chars: {total_chars:,}")
    print(f"   Est. tokens: {est_tokens/1e6:.2f}M")
    print(f"   Avg chars/line: {total_chars/len(combined):.0f}")
    
    # Save
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in combined:
            f.write(line + '\n')
    
    file_size = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"   File size: {file_size:.2f} MB")
    
    # Show samples
    print("\n🔍 Sample verification:")
    print("-" * 60)
    
    # Show 2 narrative and 2 dialogue samples
    narrative_samples = [l for l in combined if '<nl>' not in l][:2]
    dialogue_samples = [l for l in combined if '<nl>' in l][:2]
    
    print("\n📚 Narrative samples:")
    for i, s in enumerate(narrative_samples):
        print(f"   {i+1}. {s[:100]}...")
    
    print("\n💬 Dialogue samples:")
    for i, s in enumerate(dialogue_samples):
        preview = s[:150].replace(' <nl> ', '\n   ')
        print(f"   {i+1}. {preview}...")
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"""
To train with this data, update train_v020.py:

    data_path = Path('data/combined_training_data.txt')

Or run directly:

    python train_v020.py
    
(after modifying the data path)
""")


if __name__ == "__main__":
    mix_and_save()
