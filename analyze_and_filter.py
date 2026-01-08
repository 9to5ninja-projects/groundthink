"""
Analyze and filter the mixed training data.
Checks for slop, duplicates, encoding issues.
"""

from pathlib import Path
from collections import Counter
import hashlib
import re

DATA_FILE = Path("data/mixed_training_data.txt")
OUTPUT_FILE = Path("data/mixed_training_data_clean.txt")

# Slop phrases to filter - AI assistant markers
SLOP_PHRASES = [
    "as an ai", "i cannot fulfill", "openai", "chatgpt", 
    "i'm an ai", "i am an ai", "language model",
    "i hope this helps", "let me know if",
    "sure, here's", "certainly!", "of course!",
    "here's a python", "here's the code",
]

# Bad unicode patterns
BAD_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

def analyze_and_filter():
    if not DATA_FILE.exists():
        print(f"❌ {DATA_FILE} not found!")
        return
    
    print(f"🔍 Analyzing {DATA_FILE}...")
    
    with open(DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    print(f"📊 Total samples: {len(lines):,}")
    
    # Stats
    lengths = [len(l) for l in lines]
    print(f"   Avg length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"   Min length: {min(lengths)}")
    print(f"   Max length: {max(lengths)}")
    
    # Check for issues
    slop_count = 0
    bad_char_count = 0
    short_count = 0
    dupes = set()
    dupe_count = 0
    
    clean_lines = []
    slop_examples = []
    
    for i, line in enumerate(lines):
        # Check length
        if len(line) < 50:
            short_count += 1
            continue
        
        # Check for slop
        line_lower = line.lower()
        is_slop = False
        for phrase in SLOP_PHRASES:
            if phrase in line_lower:
                slop_count += 1
                if len(slop_examples) < 5:
                    slop_examples.append((phrase, line[:100]))
                is_slop = True
                break
        
        if is_slop:
            continue
        
        # Check for bad chars
        if BAD_CHARS.search(line):
            bad_char_count += 1
            line = BAD_CHARS.sub('', line)
        
        # Check for duplicates (hash first 200 chars)
        hash_key = hashlib.md5(line[:200].encode()).hexdigest()
        if hash_key in dupes:
            dupe_count += 1
            continue
        dupes.add(hash_key)
        
        clean_lines.append(line)
    
    print(f"\n🕵️  Issues Found:")
    print(f"   - Slop phrases: {slop_count}")
    print(f"   - Bad chars: {bad_char_count}")
    print(f"   - Too short: {short_count}")
    print(f"   - Duplicates: {dupe_count}")
    
    if slop_examples:
        print(f"\n⚠️  Slop examples:")
        for phrase, example in slop_examples[:3]:
            print(f"   [{phrase}]: {example}...")
    
    # Stats on clean data
    print(f"\n✅ Clean samples: {len(clean_lines):,} ({100*len(clean_lines)/len(lines):.1f}%)")
    clean_chars = sum(len(l) for l in clean_lines)
    print(f"   Total chars: {clean_chars:,}")
    print(f"   Est. tokens: ~{clean_chars//4:,}")
    
    # Save clean data
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in clean_lines:
            f.write(line + '\n')
    
    print(f"\n💾 Saved clean data to: {OUTPUT_FILE}")
    
    # Visual samples
    print(f"\n👀 Random samples from clean data:")
    import random
    for i in range(3):
        sample = random.choice(clean_lines)
        print(f"\n--- Sample {i+1} ---")
        print(sample[:300] + "..." if len(sample) > 300 else sample)

if __name__ == "__main__":
    analyze_and_filter()
