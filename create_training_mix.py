"""
Create Final Training Mix for GroundThink V2
Combines narrative data with dialogue data at target ratio.

NO DUPLICATION - we now have enough unique data!

Data sources:
- Narrative: ~31M tokens (TinyStories + Gutenberg)
- Dialogue: ~14M tokens (PersonaChat + Prosocial + DailyDialog + TV)

Target: 30% dialogue ratio
"""

import random
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
DIALOGUE_RATIO = 0.30  # 30% dialogue, 70% narrative

# Input files
NARRATIVE_FILE = DATA_DIR / "narrative_data_clean.txt"  # From earlier prep
DIALOGUE_FILE = DATA_DIR / "all_dialogue_data.txt"

# Check for alternative narrative file locations
NARRATIVE_ALTERNATIVES = [
    DATA_DIR / "training_data_clean.txt",
    DATA_DIR / "mixed_training_data_clean.txt",
    Path("groundthink/data/training_data_clean.txt"),
]

def estimate_tokens(text):
    """Estimate tokens from text (rough: 4 chars per token)."""
    return len(text) // 4

def load_data(filepath, name):
    """Load data from file."""
    if not filepath.exists():
        print(f"Warning: {name} file not found: {filepath}")
        return []
    
    lines = []
    total_chars = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                total_chars += len(line)
    
    tokens = estimate_tokens(" ".join(lines))
    print(f"Loaded {name}: {len(lines):,} samples, ~{tokens:,} tokens")
    return lines

def main():
    print("="*60)
    print("CREATING FINAL TRAINING MIX")
    print("GroundThink V2 - 30% Dialogue / 70% Narrative")
    print("="*60)
    
    # Find narrative data
    narrative_file = None
    for candidate in [NARRATIVE_FILE] + NARRATIVE_ALTERNATIVES:
        if candidate.exists():
            narrative_file = candidate
            break
    
    if narrative_file is None:
        print("\nERROR: No narrative data file found!")
        print("Please run prepare_narrative_data.py first")
        print("Checked locations:")
        for loc in [NARRATIVE_FILE] + NARRATIVE_ALTERNATIVES:
            print(f"  - {loc}")
        return
    
    # Load data
    print()
    narrative_data = load_data(narrative_file, "Narrative")
    dialogue_data = load_data(DIALOGUE_FILE, "Dialogue")
    
    if not narrative_data:
        print("ERROR: No narrative data loaded!")
        return
    if not dialogue_data:
        print("ERROR: No dialogue data loaded!")
        return
    
    # Calculate mix
    # For 30% dialogue ratio: dialogue_count / total = 0.30
    # dialogue_count / (narrative_count + dialogue_count) = 0.30
    # dialogue_count = 0.30 * narrative_count / 0.70
    
    narrative_count = len(narrative_data)
    max_dialogue = len(dialogue_data)
    
    # How much dialogue for target ratio?
    target_dialogue = int(DIALOGUE_RATIO * narrative_count / (1 - DIALOGUE_RATIO))
    
    # But don't exceed what we have!
    actual_dialogue = min(target_dialogue, max_dialogue)
    
    # Sample dialogue if we have more than needed
    if actual_dialogue < max_dialogue:
        print(f"\nSampling {actual_dialogue:,} dialogues from {max_dialogue:,} available")
        dialogue_sample = random.sample(dialogue_data, actual_dialogue)
    else:
        print(f"\nUsing all {max_dialogue:,} dialogues")
        dialogue_sample = dialogue_data
    
    # Calculate actual ratio
    total = narrative_count + len(dialogue_sample)
    actual_ratio = len(dialogue_sample) / total
    
    print(f"\nMix statistics:")
    print(f"  Narrative samples: {narrative_count:,}")
    print(f"  Dialogue samples: {len(dialogue_sample):,}")
    print(f"  Total samples: {total:,}")
    print(f"  Actual dialogue ratio: {actual_ratio:.1%}")
    
    # Combine and shuffle
    print("\nCombining and shuffling...")
    all_data = narrative_data + dialogue_sample
    random.shuffle(all_data)
    
    # Write output
    output_file = DATA_DIR / "final_training_mix.txt"
    total_chars = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_data:
            f.write(line + '\n')
            total_chars += len(line)
    
    est_tokens = total_chars // 4
    
    print(f"\nOutput written to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {est_tokens:,} (~{est_tokens/1_000_000:.1f}M)")
    
    # Show sample
    print("\n" + "="*60)
    print("Sample entries (first 3):")
    print("="*60)
    for i, line in enumerate(all_data[:3]):
        preview = line[:200] + "..." if len(line) > 200 else line
        print(f"\n[{i+1}] {preview}")
    
    print("\n" + "="*60)
    print("DONE! Training mix ready.")
    print("="*60)

if __name__ == "__main__":
    main()
