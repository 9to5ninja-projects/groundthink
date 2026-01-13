#!/usr/bin/env python3
"""Test suite for GRUArbiter (Task 0.1).

Run: python tests/test_arbiter_gru.py
"""

import sys
sys.path.insert(0, '/home/m_tes/groundthink')

import torch
from ops.arbiter_gru import GRUArbiter


def test_arbiter_shape():
    """Verify output shapes and weight normalization."""
    B, L, D = 2, 128, 768
    arbiter = GRUArbiter(D)
    
    rwkv = torch.randn(B, L, D)
    mamba = torch.randn(B, L, D)
    
    fused, weights, hidden = arbiter(rwkv, mamba)
    
    assert fused.shape == (B, L, D), f"Expected {(B, L, D)}, got {fused.shape}"
    assert weights.shape == (B, L, 2), f"Expected {(B, L, 2)}, got {weights.shape}"
    assert hidden.shape == (B, D), f"Expected {(B, D)}, got {hidden.shape}"
    
    # Weights must sum to 1.0
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(B, L)), "Weights don't sum to 1"
    print("✓ Shape test passed")


def test_arbiter_statefulness():
    """Verify GRU maintains state across calls."""
    B, L, D = 1, 10, 512
    arbiter = GRUArbiter(D)
    
    x1 = torch.randn(B, L, D)
    x2 = torch.randn(B, L, D)
    
    # First call
    _, _, h1 = arbiter(x1, x2, hidden=None)
    
    # Second call with state
    _, _, h2 = arbiter(x1, x2, hidden=h1)
    
    # Hidden states should differ (proves statefulness)
    assert not torch.allclose(h1, h2), "GRU state not updating"
    print("✓ Statefulness test passed")


def test_arbiter_zero_init():
    """Verify zero-init preserves variance."""
    arbiter = GRUArbiter(768)
    
    # Check to_weights initialized to zero
    assert torch.allclose(arbiter.to_weights.weight, torch.zeros_like(arbiter.to_weights.weight))
    assert torch.allclose(arbiter.to_weights.bias, torch.zeros_like(arbiter.to_weights.bias))
    print("✓ Zero-init test passed")


def test_arbiter_initial_weights():
    """Verify zero-init produces balanced 0.5/0.5 weights initially."""
    B, L, D = 2, 8, 256
    arbiter = GRUArbiter(D)
    
    rwkv = torch.randn(B, L, D)
    mamba = torch.randn(B, L, D)
    
    _, weights, _ = arbiter(rwkv, mamba)
    
    # With zero-init projection, softmax([0, 0]) = [0.5, 0.5]
    # But GRU hidden state evolves, so check first timestep is close to balanced
    first_weights = weights[:, 0, :]  # (B, 2)
    expected = torch.tensor([[0.5, 0.5]] * B)
    
    # Should be close to balanced at start (within tolerance due to GRU)
    assert torch.allclose(first_weights, expected, atol=0.1), \
        f"Initial weights not balanced: {first_weights}"
    print("✓ Initial weights test passed")


def test_arbiter_gradient_flow():
    """Verify gradients flow through arbiter."""
    B, L, D = 2, 16, 128
    arbiter = GRUArbiter(D)
    
    rwkv = torch.randn(B, L, D, requires_grad=True)
    mamba = torch.randn(B, L, D, requires_grad=True)
    
    fused, weights, hidden = arbiter(rwkv, mamba)
    
    # Backward pass
    loss = fused.sum()
    loss.backward()
    
    # Check gradients exist
    assert rwkv.grad is not None, "No gradient to rwkv"
    assert mamba.grad is not None, "No gradient to mamba"
    assert arbiter.to_weights.weight.grad is not None, "No gradient to to_weights"
    
    # Check gradients are non-zero
    assert rwkv.grad.abs().sum() > 0, "Zero gradient to rwkv"
    assert mamba.grad.abs().sum() > 0, "Zero gradient to mamba"
    print("✓ Gradient flow test passed")


def test_arbiter_determinism():
    """Verify same input produces same output."""
    B, L, D = 2, 16, 128
    arbiter = GRUArbiter(D)
    arbiter.eval()  # Disable dropout
    
    torch.manual_seed(42)
    rwkv = torch.randn(B, L, D)
    mamba = torch.randn(B, L, D)
    
    fused1, weights1, hidden1 = arbiter(rwkv, mamba)
    fused2, weights2, hidden2 = arbiter(rwkv, mamba)
    
    assert torch.allclose(fused1, fused2), "Non-deterministic output"
    assert torch.allclose(weights1, weights2), "Non-deterministic weights"
    assert torch.allclose(hidden1, hidden2), "Non-deterministic hidden"
    print("✓ Determinism test passed")


if __name__ == "__main__":
    test_arbiter_shape()
    test_arbiter_statefulness()
    test_arbiter_zero_init()
    test_arbiter_initial_weights()
    test_arbiter_gradient_flow()
    test_arbiter_determinism()
    print("\n✅ All Task 0.1 tests passed")
