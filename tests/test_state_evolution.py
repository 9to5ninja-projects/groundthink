"""
Linear State Evolution Test (Task 59)

Verify state responds predictably to input patterns.
Feed repeated vs varied tokens, check state changes.

Usage:
    python tests/test_state_evolution.py

Reference: VALIDATION_ROADMAP.md
Created: 2026-01-11
"""

import os
os.environ['CXX'] = '/usr/bin/g++'
os.environ['CC'] = '/usr/bin/gcc'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 16000
SEQ_LEN = 64


def state_evolution_test(model_name: str = 'GF-MH'):
    """Test: does state evolve differently for different inputs?"""
    print(f"\n{'='*50}")
    print(f" Linear State Evolution Test (Task 59)")
    print(f"{'='*50}")
    print(f" Model: {model_name} | Device: {DEVICE}")
    
    model = get_model(model_name, vocab_size=VOCAB_SIZE).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        # Test 1: Repeated token (should stabilize)
        x_repeat = torch.full((1, SEQ_LEN), 100, device=DEVICE)
        _, states_repeat = model(x_repeat, return_states=True)
        
        # Test 2: Random tokens (should vary more)
        x_random = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
        _, states_random = model(x_random, return_states=True)
        
        # Test 3: Alternating tokens
        x_alt = torch.tensor([[100, 200] * (SEQ_LEN // 2)], device=DEVICE)
        _, states_alt = model(x_alt, return_states=True)
    
    # Compare state variances
    var_repeat = states_repeat['rwkv_state'].var().item()
    var_random = states_random['rwkv_state'].var().item()
    var_alt = states_alt['rwkv_state'].var().item()
    
    print(f"\n RWKV State Variance:")
    print(f"   Repeated token: {var_repeat:.4f}")
    print(f"   Alternating:    {var_alt:.4f}")
    print(f"   Random tokens:  {var_random:.4f}")
    
    # Check if state responds to input variety
    # Random should have higher variance than repeated
    responds = var_random > var_repeat
    status = 'PASS' if responds else 'WARN'
    
    print(f"\n Result: {status}")
    print(f"   State responds to input variety: {responds}")
    print(f"{'='*50}\n")
    
    return {
        'var_repeat': var_repeat,
        'var_random': var_random,
        'var_alt': var_alt,
        'responds': responds,
        'status': status,
    }


if __name__ == '__main__':
    state_evolution_test()
