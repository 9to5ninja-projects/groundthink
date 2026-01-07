"""
RWKV-7 "Goose" Model Download Script

Run this script to download the model files before running inference.
The model will be cached locally in your huggingface cache directory.
"""

from huggingface_hub import snapshot_download
import os

# Available models - uncomment the one you want to download
MODELS = {
    "tiny": "RWKV/RWKV7-Goose-World2.8-0.1B-HF",    # ~200MB, ~100M params
    "small": "RWKV/RWKV7-Goose-World2.9-0.4B-HF",   # ~800MB, ~400M params  
    "medium": "RWKV/RWKV7-Goose-World3-1.5B-HF",    # ~3GB, ~1.5B params
    "large": "RWKV/RWKV7-Goose-World3-2.9B-HF",     # ~6GB, ~2.9B params
}

def download_model(size: str = "medium"):
    """Download the specified model size."""
    if size not in MODELS:
        print(f"Invalid size. Choose from: {list(MODELS.keys())}")
        return
    
    model_id = MODELS[size]
    print(f"Downloading {size} model: {model_id}")
    print("This may take a while depending on your internet connection...")
    print()
    
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=None,  # Use default huggingface cache
        )
        print(f"\nModel downloaded successfully!")
        print(f"Cache location: {path}")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("RWKV-7 'Goose' Model Downloader")
    print("=" * 60)
    print()
    print("Available models:")
    for size, model_id in MODELS.items():
        print(f"  {size:8s} -> {model_id}")
    print()
    
    # Get model size from command line or default to medium
    size = sys.argv[1] if len(sys.argv) > 1 else "medium"
    
    print(f"Selected size: {size}")
    print()
    
    download_model(size)
