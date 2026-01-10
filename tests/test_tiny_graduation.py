"""
Tiny Model Graduation Test Suite (Task 41)

Tests S0-S4 state space fundamentals and graduation criteria for 3.5M models.
Must pass before scaling to larger models.

Usage:
    # Activate venv first (required for mamba_ssm)
    source .venv/bin/activate
    
    python tests/test_tiny_graduation.py              # Run all tests
    python tests/test_tiny_graduation.py --states     # S0-S4 only
    python tests/test_tiny_graduation.py --gates      # G1-G4 only

Baseline Results (2026-01-10):
    S0: PASS - RWKV [1,4,32], Mamba [1,128], Gate 0.70
    S1: PASS - RWKV norm 725.7, Mamba norm 3.7
    S2: PASS - States evolve with different inputs
    S3: PASS - Deterministic (diff=0)
    S4: WARN - Variance ratio 108,583x (severe imbalance)

Reference: CANARY_TESTS.md, SCALING_MILESTONES.md, V4_TESTING.md
Created: 2026-01-10 (Task 41)
"""

import os
os.environ['CXX'] = '/usr/bin/g++'
os.environ['CC'] = '/usr/bin/gcc'

import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'GF-MH'  # Default model for graduation tests
VOCAB_SIZE = 16000    # BPE vocab size


# =============================================================================
# S0-S4: State Space Fundamentals
# =============================================================================

def test_s0_state_shapes():
    """S0: Verify state vectors exist and have correct dimensions."""
    print("\n" + "=" * 60)
    print("S0: STATE VECTOR SHAPE VERIFICATION")
    print("=" * 60)
    
    model = get_model(MODEL_NAME, vocab_size=VOCAB_SIZE).to(DEVICE)
    model.eval()
    
    x = torch.randint(0, VOCAB_SIZE, (1, 32), device=DEVICE)
    
    with torch.no_grad():
        output, states = model(x, return_states=True)
    
    # Check RWKV state exists
    assert 'rwkv_state' in states, "RWKV state missing from states dict"
    rwkv_shape = states['rwkv_state'].shape
    print(f"  RWKV state shape: {rwkv_shape}")
    
    # Check Mamba state exists
    assert 'mamba_state' in states, "Mamba state missing from states dict"
    mamba_shape = states['mamba_state'].shape
    print(f"  Mamba state shape: {mamba_shape}")
    
    # Check gate exists
    assert 'gate' in states, "Gate value missing from states dict"
    print(f"  Gate value: {states['gate']:.4f}")
    
    print(f"\n✓ S0 PASS: All state vectors present")
    return True


def test_s1_state_initialization():
    """S1: Verify states initialize to reasonable values (no NaN, bounded norms)."""
    print("\n" + "=" * 60)
    print("S1: STATE INITIALIZATION HEALTH")
    print("=" * 60)
    
    model = get_model(MODEL_NAME, vocab_size=VOCAB_SIZE).to(DEVICE)
    model.eval()
    
    x = torch.randint(0, VOCAB_SIZE, (1, 32), device=DEVICE)
    
    with torch.no_grad():
        output, states = model(x, return_states=True)
    
    rwkv_state = states['rwkv_state']
    mamba_state = states['mamba_state']
    
    # Check for NaN/Inf
    assert not torch.isnan(rwkv_state).any(), "RWKV state has NaN"
    assert not torch.isinf(rwkv_state).any(), "RWKV state has Inf"
    assert not torch.isnan(mamba_state).any(), "Mamba state has NaN"
    assert not torch.isinf(mamba_state).any(), "Mamba state has Inf"
    
    # Check norms are in reasonable range
    rwkv_norm = rwkv_state.norm().item()
    mamba_norm = mamba_state.norm().item()
    
    print(f"  RWKV state norm: {rwkv_norm:.4f}")
    print(f"  Mamba state norm: {mamba_norm:.4f}")
    
    # Norms should be non-zero and not exploded
    assert 0.001 < rwkv_norm < 1000, f"RWKV norm {rwkv_norm} outside [0.001, 1000]"
    assert 0.001 < mamba_norm < 1000, f"Mamba norm {mamba_norm} outside [0.001, 1000]"
    
    print(f"\n✓ S1 PASS: States healthy (no NaN, bounded norms)")
    return True


def test_s2_state_evolution():
    """S2: Verify states change with different inputs."""
    print("\n" + "=" * 60)
    print("S2: STATE EVOLUTION (Different Inputs → Different States)")
    print("=" * 60)
    
    model = get_model(MODEL_NAME, vocab_size=VOCAB_SIZE).to(DEVICE)
    model.eval()
    
    # Two different inputs
    input_a = torch.randint(0, VOCAB_SIZE, (1, 32), device=DEVICE)
    input_b = torch.randint(0, VOCAB_SIZE, (1, 32), device=DEVICE)
    
    with torch.no_grad():
        _, states_a = model(input_a, return_states=True)
        _, states_b = model(input_b, return_states=True)
    
    # States should differ for different inputs
    rwkv_diff = (states_a['rwkv_state'] - states_b['rwkv_state']).norm().item()
    mamba_diff = (states_a['mamba_state'] - states_b['mamba_state']).norm().item()
    
    print(f"  RWKV state diff: {rwkv_diff:.4f}")
    print(f"  Mamba state diff: {mamba_diff:.4f}")
    
    assert rwkv_diff > 0.01, f"RWKV states too similar: diff={rwkv_diff}"
    assert mamba_diff > 0.01, f"Mamba states too similar: diff={mamba_diff}"
    
    print(f"\n✓ S2 PASS: States evolve with different inputs")
    return True


def test_s3_state_determinism():
    """S3: Verify same input produces same state (deterministic)."""
    print("\n" + "=" * 60)
    print("S3: STATE DETERMINISM (Same Input → Same State)")
    print("=" * 60)
    
    model = get_model(MODEL_NAME, vocab_size=VOCAB_SIZE).to(DEVICE)
    model.eval()  # Disable dropout
    
    input_x = torch.randint(0, VOCAB_SIZE, (1, 32), device=DEVICE)
    
    with torch.no_grad():
        _, states_1 = model(input_x, return_states=True)
        _, states_2 = model(input_x, return_states=True)
    
    rwkv_diff = (states_1['rwkv_state'] - states_2['rwkv_state']).norm().item()
    mamba_diff = (states_1['mamba_state'] - states_2['mamba_state']).norm().item()
    
    print(f"  RWKV state diff: {rwkv_diff:.6f}")
    print(f"  Mamba state diff: {mamba_diff:.6f}")
    
    assert rwkv_diff < 1e-5, f"RWKV non-deterministic: diff={rwkv_diff}"
    assert mamba_diff < 1e-5, f"Mamba non-deterministic: diff={mamba_diff}"
    
    print(f"\n✓ S3 PASS: States deterministic (diff < 1e-5)")
    return True


def test_s4_component_contribution():
    """S4: Verify both components contribute (not dead)."""
    print("\n" + "=" * 60)
    print("S4: COMPONENT CONTRIBUTION (Both Active)")
    print("=" * 60)
    
    model = get_model(MODEL_NAME, vocab_size=VOCAB_SIZE).to(DEVICE)
    model.eval()
    
    x = torch.randint(0, VOCAB_SIZE, (1, 64), device=DEVICE)
    
    with torch.no_grad():
        _, states = model(x, return_states=True)
    
    rwkv_var = states['rwkv_state'].var().item()
    mamba_var = states['mamba_state'].var().item()
    
    print(f"  RWKV state variance: {rwkv_var:.6f}")
    print(f"  Mamba state variance: {mamba_var:.6f}")
    
    # Variance ratio (1.0 = perfectly balanced)
    ratio = rwkv_var / (mamba_var + 1e-8)
    print(f"  Variance ratio (RWKV/Mamba): {ratio:.2f}")
    
    # Check neither component is dead
    assert rwkv_var > 1e-6, f"RWKV state dead (var={rwkv_var})"
    assert mamba_var > 1e-6, f"Mamba state dead (var={mamba_var})"
    
    # Report balance status
    if ratio > 100 or ratio < 0.01:
        print(f"\n⚠ S4 WARN: Severe imbalance (ratio={ratio:.2f})")
        print("  Both components active but one dominates heavily")
    elif ratio > 10 or ratio < 0.1:
        print(f"\n⚠ S4 WARN: Moderate imbalance (ratio={ratio:.2f})")
    else:
        print(f"\n✓ S4 PASS: Good balance (ratio={ratio:.2f})")
    
    return {'rwkv_var': rwkv_var, 'mamba_var': mamba_var, 'ratio': ratio}


# =============================================================================
# Main Entry Point
# =============================================================================

def run_state_tests():
    """Run all S0-S4 state space tests."""
    print("\n" + "=" * 70)
    print("RUNNING S0-S4 STATE SPACE FUNDAMENTALS")
    print("=" * 70)
    
    results = {}
    
    try:
        results['S0'] = test_s0_state_shapes()
    except Exception as e:
        print(f"\n✗ S0 FAIL: {e}")
        results['S0'] = False
    
    try:
        results['S1'] = test_s1_state_initialization()
    except Exception as e:
        print(f"\n✗ S1 FAIL: {e}")
        results['S1'] = False
    
    try:
        results['S2'] = test_s2_state_evolution()
    except Exception as e:
        print(f"\n✗ S2 FAIL: {e}")
        results['S2'] = False
    
    try:
        results['S3'] = test_s3_state_determinism()
    except Exception as e:
        print(f"\n✗ S3 FAIL: {e}")
        results['S3'] = False
    
    try:
        s4_result = test_s4_component_contribution()
        results['S4'] = s4_result
    except Exception as e:
        print(f"\n✗ S4 FAIL: {e}")
        results['S4'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("S0-S4 SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v and v is not False)
    total = len(results)
    
    for test, result in results.items():
        if result is False:
            print(f"  {test}: ✗ FAIL")
        elif isinstance(result, dict):
            print(f"  {test}: ✓ (ratio={result['ratio']:.2f})")
        else:
            print(f"  {test}: ✓ PASS")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Model Graduation Tests")
    parser.add_argument('--states', action='store_true', help='Run S0-S4 state tests only')
    parser.add_argument('--gates', action='store_true', help='Run G1-G4 gate tests only')
    parser.add_argument('--model', default='GF-MH', help='Model name to test')
    args = parser.parse_args()
    
    MODEL_NAME = args.model
    print(f"Testing model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    
    if args.states or (not args.states and not args.gates):
        run_state_tests()
    
    if args.gates:
        print("\nG1-G4 gate tests not yet implemented (TODO)")
