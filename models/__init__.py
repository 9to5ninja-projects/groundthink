"""
GroundThink Model Registry

Central entry point for all model variants. No more manual import edits.

Usage:
    from models import get_model, list_models
    
    # Get a model by name
    model = get_model('8M', vocab_size=97)
    
    # List available models
    print(list_models())

Created: 2026-01-10 (Task 18.1)
"""

import sys
from pathlib import Path

# Import all factory functions from model files (now in same directory)
from .hybrid_v4 import create_hybrid_1m, create_hybrid_5m, create_hybrid_8m as create_hybrid_8m_base
from .hybrid_v4_GF import create_hybrid_GF_5m
from .hybrid_v4_WS import create_hybrid_WS_5m
from .hybrid_v4_RF import create_hybrid_RF_5m
from .hybrid_v4_CP import create_hybrid_CP_5m
from .hybrid_v4_ratio import create_hybrid_GF_RH_5m, create_hybrid_GF_MH_5m
from .hybrid_v4_8m import create_hybrid_GF_MH_8m


# Model registry - maps user-friendly names to factory functions
REGISTRY = {
    # Scale variants (baseline HY architecture)
    '1M': create_hybrid_1m,
    '5M': create_hybrid_5m,
    
    # Fusion variants (5M scale)
    'HY': create_hybrid_5m,          # Baseline per-channel gains
    'GF': create_hybrid_GF_5m,       # Gated Fusion (Phase 2 fusion winner)
    'WS': create_hybrid_WS_5m,       # Weighted Sum
    'RF': create_hybrid_RF_5m,       # Residual Fusion
    'CP': create_hybrid_CP_5m,       # Concat+Project
    
    # Ratio variants (5M scale, GF fusion)
    'GF-RH': create_hybrid_GF_RH_5m,  # RWKV-Heavy (gate init 0.7)
    'GF-MH': create_hybrid_GF_MH_5m,  # Mamba-Heavy (gate init 0.3) - Phase 2 WINNER
    
    # Scaled variants
    '8M': create_hybrid_GF_MH_8m,     # 8M GF-MH (Phase 3)
}


def get_model(name: str, vocab_size: int = 97, **kwargs):
    """
    Get a model instance by name.
    
    Args:
        name: Model name from registry (case-insensitive)
        vocab_size: Vocabulary size (default 97 for Shakespeare char-level)
        **kwargs: Additional arguments passed to factory function
        
    Returns:
        Instantiated model ready for training/inference
        
    Raises:
        ValueError: If model name not found in registry
        
    Example:
        model = get_model('8M', vocab_size=97)
        model = get_model('GF-MH', vocab_size=97)
    """
    name_upper = name.upper()
    
    if name_upper not in REGISTRY:
        available = list(REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{name}'\n"
            f"Available models: {available}\n"
            f"Hint: Use '8M' for scaled model, 'GF-MH' for Phase 2 winner"
        )
    
    factory = REGISTRY[name_upper]
    return factory(vocab_size=vocab_size, **kwargs)


def list_models():
    """
    List all available models with descriptions.
    
    Returns:
        dict: Model names mapped to brief descriptions
    """
    return {
        '1M': 'Tiny model for quick tests',
        '5M': 'Base HY model (per-channel gains)',
        'HY': 'Alias for 5M',
        'GF': 'Gated Fusion 5M (Phase 2 fusion winner)',
        'WS': 'Weighted Sum 5M',
        'RF': 'Residual Fusion 5M',
        'CP': 'Concat+Project 5M',
        'GF-RH': 'Gated Fusion RWKV-Heavy 5M (gate 0.7)',
        'GF-MH': 'Gated Fusion Mamba-Heavy 5M (gate 0.3) - PHASE 2 WINNER',
        '8M': 'Gated Fusion Mamba-Heavy 8M (Phase 3)',
    }


def get_model_info(name: str):
    """
    Get detailed info about a model.
    
    Args:
        name: Model name from registry
        
    Returns:
        dict with model metadata (params, architecture, etc.)
    """
    import torch
    
    name_upper = name.upper()
    if name_upper not in REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    
    # Create model to count params
    model = get_model(name, vocab_size=97)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'name': name_upper,
        'description': list_models().get(name_upper, 'No description'),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_M': total_params / 1e6,
    }


# Quick test when run directly
if __name__ == '__main__':
    print("=== Model Registry ===\n")
    
    print("Available models:")
    for name, desc in list_models().items():
        print(f"  {name:8} - {desc}")
    
    print("\n--- Quick validation ---")
    for name in ['5M', 'GF-MH', '8M']:
        try:
            info = get_model_info(name)
            print(f"✓ {name}: {info['params_M']:.2f}M params")
        except Exception as e:
            print(f"✗ {name}: {e}")
