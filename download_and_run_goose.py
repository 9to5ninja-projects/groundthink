"""
RWKV-7 "Goose" Model Download and Inference Script

This script downloads the RWKV-7 Goose model from Hugging Face and runs inference.
RWKV-7 is a linear-time, constant-space RNN with transformer-level performance.

Uses the native rwkv pip package which works on Windows without triton.
"""

import os

# Fix OpenMP duplicate library issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Enable RWKV-7 mode - MUST be set before importing rwkv
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"  # Set to "1" if you have compiled CUDA kernels

import torch
from huggingface_hub import hf_hub_download

# ============================================
# Configuration
# ============================================

# Available RWKV-7 Goose models (from smallest to largest):
# - "rwkv7-g1a-0.1b-20250728-ctx4096.pth"  (~100M params, 382 MB)
# - "rwkv7-g1a-0.4b-20250905-ctx4096.pth"  (~400M params, 902 MB)
# - "rwkv7-g1b-1.5b-20251202-ctx8192.pth"  (~1.5B params, 3.06 GB)
# - "rwkv7-g1c-2.9b-20251231-ctx8192.pth"  (~2.9B params, 5.9 GB) - LATEST
# - "rwkv7-g1c-7.2b-20251231-ctx8192.pth"  (~7.2B params, 14.4 GB)
# - "rwkv7-g1c-13.3b-20251231-ctx8192.pth" (~13.3B params, 26.5 GB)

MODEL_FILENAME = "rwkv7-g1c-2.9b-20251231-ctx8192.pth"  # Using the latest 2.9B model
REPO_ID = "BlinkDL/rwkv7-g1"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRATEGY = "cuda fp16" if DEVICE == "cuda" else "cpu fp32"

# ============================================
# Download Model
# ============================================

def download_model():
    """Download the RWKV-7 model from Hugging Face."""
    print(f"Downloading model: {MODEL_FILENAME}")
    print("This may take a while for large models...")
    
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        local_dir="./models"
    )
    
    print(f"Model downloaded to: {model_path}")
    return model_path

# ============================================
# Load Model
# ============================================

def load_model(model_path: str):
    """Load the RWKV-7 model using the native rwkv package."""
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE
    
    print(f"Loading model from: {model_path}")
    print(f"Using strategy: {STRATEGY}")
    
    # Load model - use forward slashes even on Windows
    # The rwkv library adds .pth automatically, so we need to strip it if present
    model_path = model_path.replace("\\", "/")
    if model_path.endswith(".pth"):
        model_path = model_path[:-4]  # Remove .pth extension
    
    model = RWKV(model=model_path, strategy=STRATEGY)
    
    # Create pipeline with the World tokenizer (for "g" and "world" models)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    
    print("Model loaded successfully!")
    return model, pipeline

# ============================================
# Inference Functions
# ============================================

def chat(pipeline, prompt: str, max_tokens: int = 512, temperature: float = 1.0, top_p: float = 0.5):
    """Generate a response to a chat prompt."""
    from rwkv.utils import PIPELINE_ARGS
    
    # Format as chat
    ctx = f"""User: {prompt}

A:"""
    
    # Generation parameters
    args = PIPELINE_ARGS(
        temperature=temperature,
        top_p=top_p,
        top_k=100,
        alpha_frequency=0.25,
        alpha_presence=0.25,
        alpha_decay=0.996,
        token_ban=[],
        token_stop=[0],  # Stop at end of text
        chunk_len=256
    )
    
    response = []
    
    def callback(token):
        response.append(token)
        print(token, end="", flush=True)
    
    pipeline.generate(ctx, token_count=max_tokens, args=args, callback=callback)
    
    return "".join(response)

def generate_text(pipeline, prompt: str, max_tokens: int = 256, temperature: float = 1.0, top_p: float = 0.7):
    """Generate text continuation (non-chat mode)."""
    from rwkv.utils import PIPELINE_ARGS
    
    args = PIPELINE_ARGS(
        temperature=temperature,
        top_p=top_p,
        top_k=100,
        alpha_frequency=0.25,
        alpha_presence=0.25,
        alpha_decay=0.996,
        token_ban=[],
        token_stop=[],
        chunk_len=256
    )
    
    response = []
    
    def callback(token):
        response.append(token)
        print(token, end="", flush=True)
    
    pipeline.generate(prompt, token_count=max_tokens, args=args, callback=callback)
    
    return "".join(response)

# ============================================
# Main Interactive Loop
# ============================================

def main():
    print("=" * 60)
    print("RWKV-7 'Goose' Model - Interactive Demo")
    print("=" * 60)
    print()
    print(f"Model: {MODEL_FILENAME}")
    print(f"Device: {DEVICE}")
    print()
    
    # Download model if needed
    model_path = download_model()
    
    # Load model
    model, pipeline = load_model(model_path)
    
    print()
    print("=" * 60)
    print("Model ready! Type your prompts below.")
    print("Type 'quit' to exit, 'mode' to switch between chat/completion")
    print("=" * 60)
    print()
    
    mode = "chat"  # Start in chat mode
    
    while True:
        try:
            user_input = input(f"[{mode}] You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'mode':
                mode = "completion" if mode == "chat" else "chat"
                print(f"Switched to {mode} mode")
                continue
            elif not user_input:
                continue
            
            print("\nAssistant: ", end="")
            
            if mode == "chat":
                response = chat(pipeline, user_input)
            else:
                response = generate_text(pipeline, user_input)
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
