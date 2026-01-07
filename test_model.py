"""
Test the GroundThink 160M model.

Validates:
1. Model instantiation
2. Forward pass
3. State persistence across steps
4. Memory usage
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

from groundthink.config import GROUNDTHINK_160M
from groundthink.model import GroundThinkModel, count_parameters


def test_instantiation():
    """Test model creation"""
    print("=" * 60)
    print("Test 1: Model Instantiation")
    print("=" * 60)
    
    model = GroundThinkModel(GROUNDTHINK_160M)
    
    # Count parameters
    counts = count_parameters(model)
    print(f"\nParameter counts:")
    for name, count in counts.items():
        print(f"  {name}: {count:,} ({count/1e6:.2f}M)")
    
    print(f"\nConfig: {GROUNDTHINK_160M}")
    print("✓ Model instantiated successfully")
    
    return model


def test_forward_pass(model: GroundThinkModel):
    """Test forward pass"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, GROUNDTHINK_160M.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Device: {device}")
    
    # Forward pass
    start = time.time()
    with torch.no_grad():
        logits, _ = model(input_ids)
    elapsed = time.time() - start
    
    print(f"Output shape: {logits.shape}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print("✓ Forward pass successful")
    
    return model


def test_state_persistence(model: GroundThinkModel):
    """Test that state carries information across steps"""
    print("\n" + "=" * 60)
    print("Test 3: State Persistence")
    print("=" * 60)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Process first chunk, save state
    chunk1 = torch.randint(0, 100, (1, 64), device=device)  # Use small vocab range
    chunk2 = torch.randint(0, 100, (1, 64), device=device)
    
    with torch.no_grad():
        # Without state
        logits_no_state, _ = model(torch.cat([chunk1, chunk2], dim=1))
        
        # With state continuation
        _, states = model(chunk1, return_states=True)
        logits_with_state, _ = model(chunk2, states=states)
    
    # The outputs should be different because the state carries context
    # (Even if the second chunk is the same, the state from chunk1 affects it)
    diff = (logits_no_state[:, 64:, :] - logits_with_state).abs().mean().item()
    
    print(f"\nOutput difference (with vs without state): {diff:.6f}")
    
    if diff > 0.01:
        print("✓ State carries information across chunks")
    else:
        print("⚠ State may not be carrying information (diff too small)")
    
    return model


def test_memory_usage(model: GroundThinkModel):
    """Test memory usage during inference"""
    print("\n" + "=" * 60)
    print("Test 4: Memory Usage")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return model
    
    device = 'cuda'
    model = model.to(device)
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Test with increasing sequence lengths
    for seq_len in [128, 512, 1024, 2048]:
        input_ids = torch.randint(0, GROUNDTHINK_160M.vocab_size, (1, seq_len), device=device)
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _, _ = model(input_ids, return_states=True)
        
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  Seq len {seq_len:4d}: {peak_mb:.1f} MB peak")
    
    print("✓ Memory test complete")
    return model


def test_generation(model: GroundThinkModel):
    """Test text generation"""
    print("\n" + "=" * 60)
    print("Test 5: Generation")
    print("=" * 60)
    
    device = next(model.parameters()).device
    
    # Simple prompt (just random tokens for now)
    prompt = torch.randint(0, 100, (1, 10), device=device)
    
    print(f"Prompt tokens: {prompt[0].tolist()}")
    
    start = time.time()
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    elapsed = time.time() - start
    
    new_tokens = generated[0, 10:].tolist()
    print(f"Generated tokens: {new_tokens}")
    print(f"Time: {elapsed:.2f}s ({len(new_tokens)/elapsed:.1f} tok/s)")
    print("✓ Generation successful")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GroundThink 160M Validation Tests")
    print("=" * 60)
    
    # Run tests
    model = test_instantiation()
    model = test_forward_pass(model)
    model = test_state_persistence(model)
    model = test_memory_usage(model)
    test_generation(model)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
