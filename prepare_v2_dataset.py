"""
GroundThink V2 Dataset Preparation (The "Editor's Cut")
=======================================================
Goal: Create a clean, deterministic, high-quality dataset for 125M Scientific Run.
Hypothesis: 125M is small. It needs HIGH DENSITY signal. No trash.

The 4 Pillars:
1. Logic (40%): FineWeb-Edu (High educational value traces)
2. Memory (30%): PG19 (Long-context books > 4096 chars)
3. Chat (20%): OpenHermes-2.5 (High quality instruction/dialog)
4. Grounding (10%): TinyStories (Fluent, simple grammar)
"""

import os
import re
import hashlib
from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
from tqdm import tqdm

# Configuration
OUTPUT_DIR = "groundthink_v2_dataset"
TOTAL_SAMPLES = 200_000 
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
    
    seen_hashes = set() # Deduplication Memory

    def is_unique(text):
        # Hash first 200 chars (header) + last 200 chars (footer)
        # Sufficient to catch copy-pasted articles
        sig = text[:200] + text[-200:]
        h = hashlib.md5(sig.encode('utf-8')).hexdigest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    def process_stream(iterator, target_count, source_name, length_filter_fn):
        samples = []
        pbar = tqdm(desc=f"Minng {source_name}", total=target_count)
        
        for item in iterator:
            raw_text = item.get('text', '')
            
            # 1. Basic Length Filter (Architecture requirement)
            if not length_filter_fn(raw_text): continue
            
            # 2. Deep Cleaning
            cleaned = clean_text(raw_text)
            if cleaned is None: continue # Rejected by cleaner
            
            # 3. Deduplication
            if not is_unique(cleaned): continue
            
            samples.append({'text': cleaned, 'source': source_name})
            pbar.update(1)
            
            if len(samples) >= target_count:
                break
        return samples

    # ---------------------------------------------------------
    # 1. LOGIC: FineWeb-Edu (The Textbook)
    # ---------------------------------------------------------
    print("1️⃣  Fetching FineWeb-Edu (Logic)...")
    ds_logic = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    # Target 40%
    logic_samples = process_stream(
        ds_logic, 
        int(TOTAL_SAMPLES * 0.4), 
        "fineweb-edu",
        lambda t: len(t) > 1000
    )

    # ---------------------------------------------------------
    # 2. MEMORY: PG19 (The Novel)
    # ---------------------------------------------------------
    print("2️⃣  Fetching PG19 (Deep Memory)...")
    ds_memory = load_dataset("pg19", split="train", streaming=True)
    # We need to manually fix PG19 extraction since it's nested
    # Wrapping iterator to normalize text field
    def pg19_wrapper(ds):
        for x in ds:
            # Strip Gutenberg Header
            t = x['text']
            if "Project Gutenberg" in t[:500]:
                start = t.find("*** START")
                if start != -1: t = t[start+100:]
            yield {'text': t[:50000]} # Truncate to 50k chars max
            
    memory_samples = process_stream(
        pg19_wrapper(ds_memory),
        int(TOTAL_SAMPLES * 0.3),
        "pg19",
        lambda t: len(t) > 8000
    )

    # ---------------------------------------------------------
    # 3. CHAT: OpenHermes-2.5 (The Persona)
    # ---------------------------------------------------------
    print("3️⃣  Fetching OpenHermes (Conversational)...")
    ds_chat = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    
    def chat_wrapper(ds):
        for x in ds:
            conv = ""
            for msg in x['conversations']:
                role = "User" if msg['from'] == 'human' else "Assistant"
                if msg['from'] == 'system': role = "System"
                conv += f"{role}: {msg['value']}\n\n"
            yield {'text': conv}

    chat_samples = process_stream(
        chat_wrapper(ds_chat),
        int(TOTAL_SAMPLES * 0.20),
        "openhermes",
        lambda t: len(t) > 500
    )

    # ---------------------------------------------------------
    # 4. GROUNDING: TinyStories (The Anchor)
    # ---------------------------------------------------------
    print("4️⃣  Fetching TinyStories (Grounding)...")
    ds_ground = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    ground_samples = process_stream(
        ds_ground,
        int(TOTAL_SAMPLES * 0.10),
        "tinystories",
        lambda t: len(t) > 200
    )

    # ---------------------------------------------------------
    # MERGE & SHUFFLE
    # ---------------------------------------------------------
    print("🌪️  Mixing and Shuffling...")
    all_data = logic_samples + memory_samples + chat_samples + ground_samples
    
    # Create HF Dataset from list
    final_ds = Dataset.from_list(all_data)
    final_ds = final_ds.shuffle(seed=SEED)
    
    print(f"💾 Saving to {OUTPUT_DIR}...")
    final_ds.save_to_disk(OUTPUT_DIR)
    print(f"✅ Design V2 Dataset Ready. Total: {len(final_ds)} samples.")
    print("   Distribution:")
    print(f"   - Logic (FineWeb): {len(logic_samples)}")
    print(f"   - Memory (PG19): {len(memory_samples)}")
    print(f"   - Chat (Hermes): {len(chat_samples)}")
    print(f"   - Grounding: {len(ground_samples)}")

if __name__ == "__main__":
    prepare_dataset()
