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
from .hybrid_v4_ratio import create_hybrid_GF_RH_5m, create_hybrid_GF_MH_5m, create_hybrid_GF_XM_5m, create_hybrid_GF_XR_5m
from .hybrid_v4_8m import create_hybrid_GF_MH_8m
from .hybrid_v4_HGF import create_hgf_balanced, create_hgf_mamba_heavy, create_hgf_rwkv_heavy
from .gpt2 import create_gpt2_5m, create_gpt2_3m, create_gpt2_8m


# Model registry - maps user-friendly names to factory functions
# Naming: tiny/small/medium/large/xl (semantic) with param counts in descriptions
REGISTRY = {
    # === Primary names (semantic sizes) ===
    'TINY': create_hybrid_1m,          # 0.5M params - quick tests
    'SMALL': create_hybrid_5m,         # 3.6M params - Phase 1-2 baseline
    'MEDIUM': create_hybrid_GF_MH_8m,  # 7.9M params - Phase 3 scale
    
    # === Fusion variants (all at SMALL scale ~3.6M) ===
    'HY': create_hybrid_5m,            # Baseline per-channel gains
    'GF': create_hybrid_GF_5m,         # Gated Fusion (Phase 2 fusion winner)
    'WS': create_hybrid_WS_5m,         # Weighted Sum
    'RF': create_hybrid_RF_5m,         # Residual Fusion
    'CP': create_hybrid_CP_5m,         # Concat+Project
    
    # === HGF: Hybrid-Gated Fusion (per-position + per-dimension) ===
    'HGF': create_hgf_balanced,        # Balanced init (50/50)
    'HGF-MH': create_hgf_mamba_heavy,  # Mamba-Heavy init (30% RWKV)
    'HGF-RH': create_hgf_rwkv_heavy,   # RWKV-Heavy init (70% RWKV)
    
    # === Ratio variants (SMALL scale, GF fusion) ===
    'GF-RH': create_hybrid_GF_RH_5m,   # RWKV-Heavy (gate init 0.7)
    'GF-MH': create_hybrid_GF_MH_5m,   # Mamba-Heavy (gate init 0.3) - Phase 2 WINNER
    'GF-XM': create_hybrid_GF_XM_5m,   # eXtreme Mamba (gate init 0.03) - 3% RWKV
    'GF-XR': create_hybrid_GF_XR_5m,   # eXtreme RWKV (gate init 0.97) - 97% RWKV
    
    # === Legacy aliases (for backward compatibility) ===
    '1M': create_hybrid_1m,            # -> TINY
    '5M': create_hybrid_5m,            # -> SMALL  
    '8M': create_hybrid_GF_MH_8m,      # -> MEDIUM
    
    # === Baselines (for controlled comparison) ===
    'GPT2': create_gpt2_5m,            # GPT-2 baseline ~5.5M params
    'GPT2-3M': create_gpt2_3m,         # Smaller GPT-2 for quick tests
    'GPT2-8M': create_gpt2_8m,         # Larger GPT-2 for scaling
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


def list_models(show=False):
    """
    List all available models with descriptions.
    
    Args:
        show: If True, prints formatted table to console
    
    Returns:
        dict: Model names mapped to brief descriptions with param counts
    """
    models = {
        # Primary names
        'tiny':   '0.5M params | ~50MB VRAM | Quick tests, debugging',
        'small':  '3.6M params | ~200MB VRAM | Phase 1-2 baseline (HY architecture)',
        'medium': '7.9M params | ~400MB VRAM | Phase 3 scale-up (GF-MH)',
        
        # Fusion variants (all ~3.6M at small scale)
        'HY':     '3.6M | Baseline per-channel gains',
        'GF':     '3.6M | Gated Fusion (Phase 2 fusion winner)',
        'WS':     '3.6M | Weighted Sum',
        'RF':     '3.6M | Residual Fusion',
        'CP':     '3.6M | Concat+Project',
        
        # HGF: Hybrid-Gated Fusion (per-position + per-dimension)
        'HGF':    '3.6M | Hybrid-Gated Fusion (pos+dim control)',
        'HGF-MH': '3.6M | HGF Mamba-Heavy (gate 0.3)',
        'HGF-RH': '3.6M | HGF RWKV-Heavy (gate 0.7)',
        
        # GF Ratio variants
        'GF-RH':  '3.6M | Gated Fusion RWKV-Heavy (gate 0.7)',
        'GF-MH':  '3.6M | Gated Fusion Mamba-Heavy (gate 0.3) ★ PHASE 2 WINNER',
        'GF-XM':  '3.6M | eXtreme Mamba (gate 0.03) - Observation 14',
        'GF-XR':  '3.6M | eXtreme RWKV (gate 0.97) - Observation 14',
        
        # Legacy aliases
        '1M':     '→ tiny (legacy alias)',
        '5M':     '→ small (legacy alias)',
        '8M':     '→ medium (legacy alias)',
    }
    
    if show:
        print("\n📦 Available Models:")
        print("-" * 60)
        for name, desc in models.items():
            print(f"  {name:10} {desc}")
        print("-" * 60)
        print("Usage: model = get_model('medium')\n")
    
    return models


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
