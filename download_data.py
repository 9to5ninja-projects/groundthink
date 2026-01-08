"""
Simple dataset downloader using requests directly.
Avoids the datasets library torch dependency issue.
"""
import os
import json
import gzip
import requests
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("groundthink_v2_dataset_raw")
OUTPUT_DIR.mkdir(exist_ok=True)

def download_tinystories():
    """Download TinyStories dataset"""
    print("📥 Downloading TinyStories...")
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    
    output_file = OUTPUT_DIR / "tinystories.txt"
    if output_file.exists():
        print(f"   Already exists: {output_file}")
        return
    
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    
    with open(output_file, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True, desc="TinyStories") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"   ✓ Saved to {output_file}")

def download_sample_data():
    """Download a sample of FineWeb-Edu using the API"""
    print("📥 Downloading FineWeb-Edu sample...")
    
    output_file = OUTPUT_DIR / "fineweb_sample.jsonl"
    if output_file.exists():
        print(f"   Already exists: {output_file}")
        return
    
    # Use the datasets viewer API to get samples
    url = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW%2Ffineweb-edu&config=sample-10BT&split=train&offset=0&length=10000"
    
    print("   Fetching from HuggingFace API...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        rows = data.get('rows', [])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in tqdm(rows, desc="Writing"):
                text = row.get('row', {}).get('text', '')
                if text:
                    f.write(json.dumps({'text': text, 'source': 'fineweb-edu'}) + '\n')
        
        print(f"   ✓ Saved {len(rows)} samples to {output_file}")
    else:
        print(f"   ❌ Failed: {response.status_code}")

def main():
    print("=" * 50)
    print("GroundThink Dataset Downloader")
    print("=" * 50)
    
    download_tinystories()
    download_sample_data()
    
    print("\n✅ Downloads complete!")
    print(f"   Output: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()
