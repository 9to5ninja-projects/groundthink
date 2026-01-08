"""
Comprehensive Dialogue Data Preparation for GroundThink V2
Downloads and processes multiple natural conversation datasets:
1. TV Dialogue (already downloaded - ~2.3k episodes)  
2. PersonaChat (131k rows) - Natural persona-based conversations
3. Prosocial-Dialog (120k dialogues) - Multi-turn conversations
4. DailyDialog - 13k daily conversations

Goal: Get enough UNIQUE dialogue data without duplication
Target: 30% dialogue ratio in training mix
"""

import os
import json
import requests
import random
from pathlib import Path
from collections import defaultdict
import re

# Output directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_file(url, filename):
    """Download a file with progress indication."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    downloaded = 0
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r  {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB ({pct:.1f}%)", end="")
    print()
    return filename

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove excessive punctuation
    text = re.sub(r'([!?.]){3,}', r'\1', text)
    return text

def format_dialogue(turns, source="unknown"):
    """Format dialogue turns into training text.
    Uses natural conversation format, NOT User/Assistant format.
    """
    if not turns or len(turns) < 2:
        return None
    
    # Join turns with newlines
    formatted = "\n".join(clean_text(t) for t in turns if clean_text(t))
    
    if len(formatted) < 50:  # Too short
        return None
    
    return formatted

# =============================================================================
# 1. PERSONACHAT - Natural persona-based conversations
# =============================================================================

def download_personachat():
    """Download PersonaChat dataset (truecased version)."""
    print("\n" + "="*60)
    print("PERSONACHAT - 131k natural persona conversations")
    print("="*60)
    
    # PersonaChat parquet files
    base_url = "https://huggingface.co/datasets/bavard/personachat_truecased/resolve/main/data"
    
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    dialogues = []
    
    # Try to download full dataset parquet
    train_url = "https://huggingface.co/datasets/bavard/personachat_truecased/resolve/refs%2Fconvert%2Fparquet/full/train/0000.parquet"
    parquet_file = cache_dir / "personachat_train.parquet"
    
    if not parquet_file.exists():
        try:
            download_file(train_url, parquet_file)
        except Exception as e:
            print(f"Could not download parquet: {e}")
            # Try JSON alternative
            return download_personachat_json()
    
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(df)} rows from PersonaChat")
        print(f"Columns: {list(df.columns)}")
        
        # Group by conversation
        conv_groups = df.groupby('conv_id')
        
        for conv_id, group in conv_groups:
            # Sort by utterance index
            group = group.sort_values('utterance_idx')
            
            # Build dialogue from history + final response
            turns = []
            for idx, row in group.iterrows():
                history = row.get('history', [])
                candidates = row.get('candidates', [])
                
                # Handle numpy arrays
                if hasattr(history, 'tolist'):
                    history = history.tolist()
                if hasattr(candidates, 'tolist'):
                    candidates = candidates.tolist()
                
                if candidates and len(candidates) > 0:
                    response = candidates[-1]  # Last candidate is the true response
                    
                    if not turns and history and len(history) > 0:
                        turns = list(history)
                    if response:
                        turns.append(response)
            
            if len(turns) >= 2:
                formatted = format_dialogue(turns, "personachat")
                if formatted:
                    dialogues.append(formatted)
        
        print(f"Extracted {len(dialogues)} PersonaChat dialogues")
        return dialogues
        
    except Exception as e:
        print(f"Error processing PersonaChat parquet: {e}")
        return []

def download_personachat_json():
    """Fallback: Download PersonaChat from alternative source."""
    print("Trying alternative PersonaChat source...")
    return []

# =============================================================================
# 2. PROSOCIAL-DIALOG - Multi-turn ethical dialogues
# =============================================================================

def download_prosocial_dialog():
    """Download Prosocial-Dialog dataset (120k dialogues)."""
    print("\n" + "="*60)
    print("PROSOCIAL-DIALOG - 120k multi-turn conversations")
    print("="*60)
    
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    train_url = "https://huggingface.co/datasets/allenai/prosocial-dialog/resolve/main/train.json"
    json_file = cache_dir / "prosocial_train.json"
    
    if not json_file.exists():
        download_file(train_url, json_file)
    
    dialogues = []
    
    try:
        # It's JSONL format (one JSON per line), not a single JSON array
        data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except:
                        continue
        
        print(f"Loaded {len(data)} entries from Prosocial-Dialog")
        
        # Group by dialogue_id if available
        dialogue_groups = defaultdict(list)
        
        for entry in data:
            context = entry.get('context', '')
            response = entry.get('response', '')
            rot = entry.get('rots', [])  # Rules of thumb
            
            # Context often contains the conversation history with __SEP__
            if '__SEP__' in context:
                turns = context.split('__SEP__')
                turns = [t.strip() for t in turns if t.strip()]
                turns.append(response)
            else:
                turns = [context, response] if context and response else []
            
            if len(turns) >= 2:
                formatted = format_dialogue(turns, "prosocial")
                if formatted:
                    dialogues.append(formatted)
        
        # Deduplicate
        dialogues = list(set(dialogues))
        print(f"Extracted {len(dialogues)} unique Prosocial-Dialog conversations")
        return dialogues
        
    except Exception as e:
        print(f"Error processing Prosocial-Dialog: {e}")
        import traceback
        traceback.print_exc()
        return []

# =============================================================================
# 3. DAILYDIALOG - Daily life conversations
# =============================================================================

def download_dailydialog():
    """Download DailyDialog dataset."""
    print("\n" + "="*60)
    print("DAILYDIALOG - 13k daily conversations")
    print("="*60)
    
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    # DailyDialog parquet
    train_url = "https://huggingface.co/datasets/daily_dialog/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    parquet_file = cache_dir / "dailydialog_train.parquet"
    
    if not parquet_file.exists():
        try:
            download_file(train_url, parquet_file)
        except Exception as e:
            print(f"Could not download DailyDialog: {e}")
            return []
    
    dialogues = []
    
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_file)
        print(f"Loaded {len(df)} dialogues from DailyDialog")
        print(f"Columns: {list(df.columns)}")
        
        for idx, row in df.iterrows():
            dialog = row.get('dialog', [])
            # Handle numpy array
            if hasattr(dialog, 'tolist'):
                dialog = dialog.tolist()
            if isinstance(dialog, list) and len(dialog) >= 2:
                formatted = format_dialogue(dialog, "dailydialog")
                if formatted:
                    dialogues.append(formatted)
        
        dialogues = list(set(dialogues))
        print(f"Extracted {len(dialogues)} unique DailyDialog conversations")
        return dialogues
        
    except Exception as e:
        print(f"Error processing DailyDialog: {e}")
        return []

# =============================================================================
# 4. LOAD EXISTING TV DIALOGUE DATA
# =============================================================================

def load_existing_tv_dialogue():
    """Load existing TV dialogue data if available."""
    print("\n" + "="*60)
    print("TV DIALOGUE - Already processed")
    print("="*60)
    
    tv_file = DATA_DIR / "dialogue_training_data.txt"
    
    if not tv_file.exists():
        print("TV dialogue file not found, skipping...")
        return []
    
    dialogues = []
    with open(tv_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Convert <nl> markers back to newlines for consistency
                text = line.replace('<nl>', '\n')
                dialogues.append(text)
    
    print(f"Loaded {len(dialogues)} TV dialogues")
    return dialogues

# =============================================================================
# MAIN: Combine all dialogue sources
# =============================================================================

def main():
    print("="*60)
    print("COMPREHENSIVE DIALOGUE DATA PREPARATION")
    print("GroundThink V2 - Natural Conversation Training")
    print("="*60)
    
    all_dialogues = []
    source_counts = {}
    
    # 1. Load existing TV dialogue
    tv_dialogues = load_existing_tv_dialogue()
    if tv_dialogues:
        all_dialogues.extend([(d, "tv_dialogue") for d in tv_dialogues])
        source_counts["tv_dialogue"] = len(tv_dialogues)
    
    # 2. Download PersonaChat
    try:
        pc_dialogues = download_personachat()
        if pc_dialogues:
            all_dialogues.extend([(d, "personachat") for d in pc_dialogues])
            source_counts["personachat"] = len(pc_dialogues)
    except Exception as e:
        print(f"PersonaChat failed: {e}")
    
    # 3. Download Prosocial-Dialog
    try:
        ps_dialogues = download_prosocial_dialog()
        if ps_dialogues:
            all_dialogues.extend([(d, "prosocial") for d in ps_dialogues])
            source_counts["prosocial"] = len(ps_dialogues)
    except Exception as e:
        print(f"Prosocial-Dialog failed: {e}")
    
    # 4. Download DailyDialog
    try:
        dd_dialogues = download_dailydialog()
        if dd_dialogues:
            all_dialogues.extend([(d, "dailydialog") for d in dd_dialogues])
            source_counts["dailydialog"] = len(dd_dialogues)
    except Exception as e:
        print(f"DailyDialog failed: {e}")
    
    # Shuffle all dialogues
    random.shuffle(all_dialogues)
    
    # Write combined output
    output_file = DATA_DIR / "all_dialogue_data.txt"
    
    total_chars = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for dialogue, source in all_dialogues:
            # Convert newlines to <nl> markers for single-line format
            formatted = dialogue.replace('\n', '<nl>')
            f.write(formatted + '\n')
            total_chars += len(dialogue)
    
    # Estimate tokens (rough: 4 chars per token)
    est_tokens = total_chars // 4
    
    # Print summary
    print("\n" + "="*60)
    print("DIALOGUE DATA SUMMARY")
    print("="*60)
    print(f"\nTotal unique dialogues: {len(all_dialogues):,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {est_tokens:,} (~{est_tokens/1_000_000:.1f}M)")
    print(f"\nOutput file: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\nBreakdown by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_dialogues) * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")
    
    print("\nDone! Ready for training mix.")

if __name__ == "__main__":
    main()
