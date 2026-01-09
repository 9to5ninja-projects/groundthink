"""
GroundThink Dialogue Data Preparation
=====================================
Downloads and processes the tv_dialogue dataset for natural conversation training.
Format: Speaker-tagged dialogue without User/Assistant conditioning.

Target output: dialogue_training_data.txt (one conversation per line)

NOTE: Uses requests + parquet to avoid datasets library issues on Windows.
"""

import os
import re
import hashlib
import json
from pathlib import Path

# Try to import optional dependencies
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "dialogue_training_data.txt"
STATS_FILE = OUTPUT_DIR / "dialogue_stats.txt"

# Minimum dialogue length (characters) to include
MIN_LENGTH = 200
MAX_LENGTH = 8000  # Avoid very long episodes

# ==========================================
# TEXT CLEANING
# ==========================================
def clean_dialogue(text):
    """
    Clean TV/movie dialogue while preserving natural speaker format.
    Returns None if text should be rejected.
    """
    if not text or len(text) < MIN_LENGTH:
        return None
    
    if len(text) > MAX_LENGTH:
        # Truncate to reasonable length (find natural break point)
        text = text[:MAX_LENGTH]
        # Find last complete line
        last_newline = text.rfind('\n')
        if last_newline > MIN_LENGTH:
            text = text[:last_newline]
    
    # 1. ASCII Density Check (keep English content)
    try:
        ascii_count = len([c for c in text if ord(c) < 128])
        if ascii_count / len(text) < 0.90:  # Slightly more lenient for dialogue
            return None
    except:
        return None
    
    # 2. Reject if it looks like code or markup
    if text.count('{') + text.count('}') > 5:
        return None
    if '<div' in text.lower() or '<html' in text.lower():
        return None
    
    # 3. Smart quote normalization
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace('—', ' - ').replace('–', ' - ')
    text = text.replace('\u200b', '')  # zero-width space
    
    # 4. Normalize the speaker format [NAME] -> consistent
    # Already in [NAME] format from dataset, just clean up
    
    # 5. Collapse excessive whitespace within lines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # 6. Must have at least 2 speakers (it's a dialogue!)
    speaker_count = len(re.findall(r'\[([A-Z][A-Z\s]+)\]', text))
    if speaker_count < 2:
        return None
    
    # 7. Convert to single-line format (newlines -> special separator)
    # This matches your LineDataset which reads one sample per line
    text = text.replace('\n', ' <nl> ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def format_for_training(dialogue_text, show_info):
    """
    Format dialogue with optional context prefix.
    
    Input format from dataset:
    [SPEAKER1] Line 1
    [SPEAKER2] Line 2
    (scene direction)
    
    Output: Single line with <nl> as line separator
    """
    # Clean and validate
    cleaned = clean_dialogue(dialogue_text)
    if cleaned is None:
        return None
    
    return cleaned


# ==========================================
# MAIN PROCESSING
# ==========================================
def download_parquet():
    """Download the tv_dialogue parquet file directly"""
    PARQUET_URL = "https://huggingface.co/datasets/sedthh/tv_dialogue/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    LOCAL_PARQUET = OUTPUT_DIR / "tv_dialogue.parquet"
    
    if LOCAL_PARQUET.exists():
        print(f"   ✓ Using cached: {LOCAL_PARQUET}")
        return LOCAL_PARQUET
    
    if not HAS_REQUESTS:
        print("   ❌ requests not installed. Run: pip install requests")
        return None
    
    print(f"   Downloading from HuggingFace...")
    
    try:
        response = requests.get(PARQUET_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(LOCAL_PARQUET, 'wb') as f:
            if total_size > 0:
                for chunk in tqdm(response.iter_content(chunk_size=8192), 
                                  total=total_size//8192, 
                                  unit='KB', 
                                  desc="Downloading"):
                    f.write(chunk)
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"   ✓ Downloaded: {LOCAL_PARQUET.stat().st_size / 1024 / 1024:.1f} MB")
        return LOCAL_PARQUET
    
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return None


def download_and_process():
    print("=" * 60)
    print("GroundThink Dialogue Data Preparation")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # --------------------------------------------------------
    # Step 1: Download tv_dialogue dataset
    # --------------------------------------------------------
    print("\n📥 Downloading tv_dialogue dataset from HuggingFace...")
    print("   Source: sedthh/tv_dialogue")
    print("   Content: Friends, Doctor Who, Star Trek, Marvel, etc.")
    
    if not HAS_PANDAS:
        print("   ❌ pandas not installed. Run: pip install pandas pyarrow")
        return None
    
    parquet_file = download_parquet()
    if parquet_file is None:
        return None
    
    # Load with pandas
    print("\n📖 Loading parquet file...")
    df = pd.read_parquet(parquet_file)
    print(f"   ✓ Loaded {len(df):,} episodes")
    print(f"   Columns: {list(df.columns)}")
    
    # Normalize column names (HF sometimes uses uppercase)
    df.columns = [c.lower() for c in df.columns]
    
    # --------------------------------------------------------
    # Step 2: Process and clean
    # --------------------------------------------------------
    print("\n🧹 Processing dialogues...")
    
    processed = []
    rejected_short = 0
    rejected_quality = 0
    rejected_speakers = 0
    seen_hashes = set()
    
    source_counts = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = row.get('text', '')
        source = row.get('source', 'unknown')
        
        # Track source distribution
        source_key = source.split('/')[0] if '/' in source else source
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        
        # Basic length check
        if len(str(text)) < MIN_LENGTH:
            rejected_short += 1
            continue
        
        # Clean and format
        cleaned = format_for_training(str(text), source)
        
        if cleaned is None:
            rejected_quality += 1
            continue
        
        # Deduplication
        sig = cleaned[:200] + cleaned[-200:]
        h = hashlib.md5(sig.encode('utf-8')).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        processed.append(cleaned)
    
    print(f"\n   ✓ Processed: {len(processed):,} dialogues")
    print(f"   ✗ Rejected (too short): {rejected_short:,}")
    print(f"   ✗ Rejected (quality): {rejected_quality:,}")
    
    # --------------------------------------------------------
    # Step 3: Verify samples
    # --------------------------------------------------------
    print("\n🔍 VERIFICATION - Sample dialogues:")
    print("-" * 60)
    
    for i, sample in enumerate(processed[:3]):
        # Show first 500 chars of each sample
        preview = sample[:500].replace(' <nl> ', '\n')
        print(f"\n📝 Sample {i+1}:")
        print(preview)
        print("...")
        print("-" * 60)
    
    # --------------------------------------------------------
    # Step 4: Calculate statistics
    # --------------------------------------------------------
    print("\n📊 Statistics:")
    
    total_chars = sum(len(s) for s in processed)
    avg_chars = total_chars / len(processed) if processed else 0
    est_tokens = total_chars / 4  # ~4 chars per token
    
    # Count speaker turns
    total_turns = sum(s.count('[') for s in processed)
    avg_turns = total_turns / len(processed) if processed else 0
    
    print(f"   Total dialogues: {len(processed):,}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Avg chars/dialogue: {avg_chars:.0f}")
    print(f"   Est. tokens: {est_tokens/1e6:.2f}M")
    print(f"   Avg speaker turns: {avg_turns:.1f}")
    
    print(f"\n   Source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / sum(source_counts.values())
        print(f"   - {source}: {count:,} ({pct:.1f}%)")
    
    # --------------------------------------------------------
    # Step 5: Save
    # --------------------------------------------------------
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in processed:
            f.write(line + '\n')
    
    # Save stats
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write("GroundThink Dialogue Data Statistics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total dialogues: {len(processed):,}\n")
        f.write(f"Total characters: {total_chars:,}\n")
        f.write(f"Avg chars/dialogue: {avg_chars:.0f}\n")
        f.write(f"Est. tokens: {est_tokens/1e6:.2f}M\n")
        f.write(f"Avg speaker turns: {avg_turns:.1f}\n")
        f.write(f"\nSource distribution:\n")
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {source}: {count:,}\n")
    
    print(f"\n✅ Done!")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Stats: {STATS_FILE}")
    
    return processed


def verify_output():
    """Triple-check the output file"""
    print("\n" + "=" * 60)
    print("TRIPLE CHECK VERIFICATION")
    print("=" * 60)
    
    if not OUTPUT_FILE.exists():
        print("❌ Output file not found! Run download_and_process() first.")
        return
    
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n1️⃣  File Check:")
    print(f"   File exists: ✓")
    print(f"   Total lines: {len(lines):,}")
    print(f"   File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
    
    print(f"\n2️⃣  Format Check:")
    # Check that lines are properly formatted
    valid_lines = 0
    has_speakers = 0
    has_newline_markers = 0
    
    for line in lines[:100]:  # Check first 100
        line = line.strip()
        if len(line) > 0:
            valid_lines += 1
        if '[' in line and ']' in line:
            has_speakers += 1
        if '<nl>' in line:
            has_newline_markers += 1
    
    print(f"   Valid lines (of first 100): {valid_lines}")
    print(f"   Has speaker tags: {has_speakers}")
    print(f"   Has <nl> markers: {has_newline_markers}")
    
    print(f"\n3️⃣  Content Check - Random samples:")
    import random
    random.seed(42)
    samples = random.sample(lines, min(5, len(lines)))
    
    for i, sample in enumerate(samples):
        sample = sample.strip()
        # Show with newlines restored
        preview = sample[:300].replace(' <nl> ', '\n')
        print(f"\n   Sample {i+1} (first 300 chars):")
        print("   " + preview.replace('\n', '\n   '))
    
    print(f"\n4️⃣  Compatibility Check:")
    # Test that it loads with LineDataset format
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            test_lines = [line.strip() for line in f if len(line.strip()) > 50]
        print(f"   Lines > 50 chars: {len(test_lines):,}")
        print(f"   ✓ Compatible with LineDataset format")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n✅ VERIFICATION COMPLETE")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    processed = download_and_process()
    
    if processed:
        verify_output()
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("""
To use this data for training:

Option A: Train on dialogue only
    - Modify train_v020.py to use 'dialogue_training_data.txt'
    
Option B: Mix with existing data (RECOMMENDED)
    - Run: python mix_dialogue_data.py
    - This will combine dialogue + existing narrative data

Option C: Manual mixing
    - Append dialogue_training_data.txt to mixed_training_data.txt
""")
