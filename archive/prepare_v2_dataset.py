"""
GroundThink V2 Dataset Preparation (The "Editor's Cut")
=======================================================
Goal: Create a clean, deterministic, high-quality dataset for 125M Scientific Run.
Hypothesis: 125M is small. It needs HIGH DENSITY signal. No trash.

The 4 Pillars:
1. Logic (60%): FineWeb-Edu (High educational value traces)
2. Memory (30%): Cosmopedia Stories (Long-context narrative)
3. Grounding (10%): TinyStories (Fluent, simple grammar)
"""

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # Fast downloads

import re
import hashlib
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "groundthink_v2_dataset"
TOTAL_SAMPLES = 500_000  # Start smaller for local testing, scale up for A100
SEED = 42

# ==========================================
# TEXT CLEANING & FILTERS
# ==========================================
def clean_text(text):
    """
    The 'Editor's Cut' Sanitizer.
    Returns None if text should be rejected.
    Returns cleaned string if acceptable.
    """
    if not text or len(text) < 100: return None

    # 1. ASCII Density Check (Reject Non-English / Binary garbage)
    # Fast heuristic: Count ascii vs non-ascii
    try:
        # Check if 95% of chars are standard ASCII
        # This kills Chinese/Russian/Emoji-heavy spam
        ascii_count = len([c for c in text if ord(c) < 128])
        if ascii_count / len(text) < 0.95: 
            return None # Reject foreign language or heavily corrupted text
    except:
        return None

    # 2. Heuristic: Reject "Code" (Too many curly braces or HTML tags)
    if text.count('{') + text.count('}') > 10: return None
    if text.count('<div') > 0 or text.count('<html') > 0: return None

    # 3. Reject AI Slop / Refusals / Identity Errors (The "No Bullshit" Filter)
    # matching lower case for broad coverage
    text_lower = text.lower()
    slop_phrases = [
        "as an ai", "language model", "i cannot", "program", 
        "moral guidelines", "openai", "anthropic", "google", "meta",
        "copyrighted material", "offensive or harmful", "i apologize",
        "hope this helps", "feel free to ask", "unable to provide"
    ]
    if any(s in text_lower for s in slop_phrases):
        return None # Nuked

    # 4. Normalization
    # Smart Quotes -> Normal
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    # Em Dash -> Simple Dash
    text = text.replace('—', ' - ').replace('–', ' - ')
    # Zero width space
    text = text.replace('\u200b', '')
    
    # 4. Whitespace Cleanup
    # specific fix for " . " or " , " causing tokenization grief
    text = re.sub(r'\s+([.,;:?!])', r'\1', text) 
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Emoji Stripping (Basic range)
    # This regex is a coarse filter for surrogate pairs / common emoji ranges
    try:
        text = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE).sub(r'', text)
    except:
        pass # Regex failure on some systems matches nothing, acceptable fallback

    return text

def prepare_dataset():
    print(f"🧹 Starting Editor's Cut Selection (Target: {TOTAL_SAMPLES} samples)...")
    print(f"📁 Output: {OUTPUT_DIR}")
    
    seen_hashes = set() # Deduplication Memory

    def is_unique(text):
        # Hash first 200 chars (header) + last 200 chars (footer)
        sig = text[:200] + text[-200:]
        h = hashlib.md5(sig.encode('utf-8')).hexdigest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    def process_stream(iterator, target_count, source_name, length_filter_fn, text_key='text'):
        samples = []
        rejected = 0
        pbar = tqdm(desc=f"Mining {source_name}", total=target_count, unit="samples")
        
        for item in iterator:
            raw_text = item.get(text_key, '')
            
            # 1. Basic Length Filter
            if not length_filter_fn(raw_text): 
                rejected += 1
                continue
            
            # 2. Deep Cleaning
            cleaned = clean_text(raw_text)
            if cleaned is None: 
                rejected += 1
                continue
            
            # 3. Deduplication
            if not is_unique(cleaned): 
                rejected += 1
                continue
            
            samples.append({'text': cleaned, 'source': source_name})
            pbar.update(1)
            
            if len(samples) >= target_count:
                break
        
        pbar.close()
        print(f"   ✓ {source_name}: {len(samples)} samples (rejected {rejected})")
        return samples

    # ---------------------------------------------------------
    # 1. LOGIC: FineWeb-Edu (60%)
    # ---------------------------------------------------------
    print("\n1️⃣  Fetching FineWeb-Edu (Logic)...")
    try:
        ds_logic = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name="sample-10BT", 
            split="train", 
            streaming=True,
            trust_remote_code=True
        )
        logic_target = int(TOTAL_SAMPLES * 0.6)
        logic_samples = process_stream(
            ds_logic, 
            logic_target, 
            "fineweb-edu",
            lambda t: len(t) > 500  # Lower threshold for more samples
        )
    except Exception as e:
        print(f"⚠️ FineWeb-Edu failed: {e}")
        logic_samples = []

    # ---------------------------------------------------------
    # 2. MEMORY: Cosmopedia Stories (30%)
    # ---------------------------------------------------------
    print("\n2️⃣  Fetching Cosmopedia-Stories (Memory/Narrative)...")
    try:
        ds_memory = load_dataset(
            "HuggingFaceTB/cosmopedia", 
            "stories", 
            split="train", 
            streaming=True,
            trust_remote_code=True
        )
        memory_target = int(TOTAL_SAMPLES * 0.3)
        memory_samples = process_stream(
            ds_memory,
            memory_target,
            "cosmopedia-stories",
            lambda t: len(t) > 1000,
            text_key='text'
        )
    except Exception as e:
        print(f"⚠️ Cosmopedia failed: {e}. Using more FineWeb instead.")
        memory_samples = []

    # ---------------------------------------------------------
    # 3. GROUNDING: TinyStories (10%)
    # ---------------------------------------------------------
    print("\n3️⃣  Fetching TinyStories (Grounding)...")
    try:
        ds_ground = load_dataset(
            "roneneldan/TinyStories", 
            split="train", 
            streaming=True,
            trust_remote_code=True
        )
        ground_target = int(TOTAL_SAMPLES * 0.10)
        ground_samples = process_stream(
            ds_ground,
            ground_target,
            "tinystories",
            lambda t: len(t) > 100
        )
    except Exception as e:
        print(f"⚠️ TinyStories failed: {e}")
        ground_samples = []

    # ---------------------------------------------------------
    # MERGE & SHUFFLE
    # ---------------------------------------------------------
    print("\n🌪️  Mixing and Shuffling...")
    all_data = logic_samples + memory_samples + ground_samples
    
    if len(all_data) == 0:
        print("❌ No samples collected! Check network/dataset access.")
        return
    
    # Create HF Dataset from list
    final_ds = Dataset.from_list(all_data)
    final_ds = final_ds.shuffle(seed=SEED)
    
    print(f"\n💾 Saving to {OUTPUT_DIR}...")
    final_ds.save_to_disk(OUTPUT_DIR)
    
    # Calculate token estimate
    avg_chars = sum(len(x['text']) for x in all_data) / len(all_data)
    est_tokens = len(all_data) * avg_chars / 4  # ~4 chars per token
    
    print(f"\n✅ Dataset Ready!")
    print(f"   Total samples: {len(final_ds):,}")
    print(f"   Avg chars/sample: {avg_chars:.0f}")
    print(f"   Est. tokens: {est_tokens/1e6:.1f}M")
    print(f"\n   Distribution:")
    print(f"   - Logic (FineWeb): {len(logic_samples):,}")
    print(f"   - Memory (Cosmopedia): {len(memory_samples):,}")
    print(f"   - Grounding (TinyStories): {len(ground_samples):,}")

if __name__ == "__main__":
    prepare_dataset()
