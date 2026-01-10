"""
GroundThink V4 Hybrid Model (HY - Hybrid Per-Channel Fusion)

TRUE PARALLEL ARCHITECTURE: RWKV-6 and Mamba-2 run side-by-side in EVERY block.
Both kernels process the same normalized input, outputs combined with learned fusion gains.

Fusion Type: HY (Hybrid per-channel) - 256 params at hidden=128
- rwkv_gain and mamba_gain are per-dimension learnable weights

Goal: Break the 7.0 Loss Wall through gradient independence of parallel pathways.
- RWKV-6: Smooth decay memory for persona over 2048+ tokens
- Mamba-2: Selective memory for snapping to new instructions
"""

# Fix OpenMP conflict BEFORE any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# FLA library imports
from fla.layers.rwkv6 import RWKV6Attention
from fla.layers.mamba2 import Mamba2


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class ParallelHybridBlock(nn.Module):
    """
    Parallel Hybrid Block: RWKV-6 and Mamba-2 run IN PARALLEL
    
    Fusion Type: HY (Hybrid per-channel gains)
    - rwkv_gain: per-dimension weight for RWKV output
    - mamba_gain: per-dimension weight for Mamba output
    - 256 params at hidden=128
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        
        # Pre-norm for attention pathways
        self.ln = RMSNorm(hidden_size)
        
        # PARALLEL attention mechanisms
        # RWKV6: key_dim = hidden_size * 0.5 must be divisible by num_heads
        # For hidden=128, key_dim=64, so num_heads must divide 64
        self.rwkv6 = RWKV6Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_idx=layer_idx,
        )
        
        # Mamba2: num_heads = expand * hidden_size / head_dim (from Mamba2Config)
        # For hidden=128, expand=2, head_dim=64: num_heads = 2*128/64 = 4
        # We must ensure: num_heads = expand * hidden_size / head_dim
        mamba_expand = 2
        mamba_head_dim = 64
        mamba_heads = (mamba_expand * hidden_size) // mamba_head_dim
        self.mamba2 = Mamba2(
            hidden_size=hidden_size,
            num_heads=mamba_heads,
            head_dim=mamba_head_dim,
            expand=mamba_expand,
            layer_idx=layer_idx,
        )
        
        # Normalize each component's output BEFORE fusion
        # This ensures both contribute equally regardless of internal scale differences
        self.rwkv_out_norm = RMSNorm(hidden_size)
        self.mamba_out_norm = RMSNorm(hidden_size)
        
        # HY Fusion: Hybrid per-channel gains (256 params at hidden=128)
        # Init: rwkv=0.7, mamba=0.3 (sum near 1.0, slight RWKV preference)
        self.rwkv_gain = nn.Parameter(torch.ones(hidden_size) * 0.7)
        self.mamba_gain = nn.Parameter(torch.ones(hidden_size) * 0.3)
        
        # FFN with 4x expansion
        ffn_hidden = int(hidden_size * ffn_mult)
        self.ffn_ln = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size, bias=False),
        )
    
    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            return_activations: If True, also return component outputs for monitoring
        Returns:
            out: [batch_size, seq_len, hidden_size]
            activations (optional): dict with 'rwkv' and 'mamba' tensors
        """
        # Pre-norm
        norm_x = self.ln(x)
        
        # PARALLEL pathways - both see same normalized input
        # RWKV6Attention returns (output, attention_weights, past_key_values) tuple
        out_rwkv, _, _ = self.rwkv6(norm_x)
        # Mamba2 returns just the tensor
        out_mamba = self.mamba2(norm_x)
        
        # Normalize outputs to unit variance before applying gains
        # This ensures fair competition between components
        out_rwkv = self.rwkv_out_norm(out_rwkv)
        out_mamba = self.mamba_out_norm(out_mamba)
        
        # Combine with learned gains + residual
        x = x + (self.rwkv_gain * out_rwkv) + (self.mamba_gain * out_mamba)
        
        # FFN with residual
        x = x + self.ffn(self.ffn_ln(x))
        
        if return_activations:
            return x, {'rwkv': out_rwkv, 'mamba': out_mamba}
        return x


class HybridModel(nn.Module):
    """
    GroundThink V4 Hybrid Language Model
    
    Stack of ParallelHybridBlocks with shared embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 128,
        num_layers: int = 8,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # Stack of parallel hybrid blocks
        self.blocks = nn.ModuleList([
            ParallelHybridBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])
        
        # Final norm and output projection
        self.ln_out = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie embeddings (saves params, helps small models)
        if tie_embeddings:
            self.head.weight = self.embed.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """
        Args:
            x: Token IDs [batch_size, seq_len]
            return_activations: If True, also return per-layer component activations
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            all_activations (optional): list of dicts, one per layer
        """
        # Embed tokens
        h = self.embed(x)
        
        all_activations = [] if return_activations else None
        
        # Pass through all blocks
        for block in self.blocks:
            if return_activations:
                h, acts = block(h, return_activations=True)
                all_activations.append(acts)
            else:
                h = block(h)
        
        # Final norm and project to vocab
        h = self.ln_out(h)
        logits = self.head(h)
        
        if return_activations:
            return logits, all_activations
        return logits
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters, optionally excluding embeddings"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
        return n_params


def create_hybrid_1m(vocab_size: int = 10000) -> HybridModel:
    """Create ~500K parameter hybrid model for fast iteration testing"""
    return HybridModel(
        vocab_size=vocab_size,
        hidden_size=64,
        num_layers=4,
        num_heads=2,
        ffn_mult=4.0,
    )


def create_hybrid_5m(vocab_size: int = 10000) -> HybridModel:
    """Create ~5M parameter hybrid model"""
    return HybridModel(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        ffn_mult=4.0,
    )


def create_hybrid_8m(vocab_size: int = 10000) -> HybridModel:
    """Create ~8M parameter hybrid model"""
    return HybridModel(
        vocab_size=vocab_size,
        hidden_size=192,
        num_layers=12,
        num_heads=6,
        ffn_mult=4.0,
    )


# Quick test
if __name__ == "__main__":
    print("Testing HybridModel...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_hybrid_5m(vocab_size=256).to(device)  # Small vocab for quick test
    print(f"Model created: {model.get_num_params():,} non-embedding params")
    
    # Test forward pass
    x = torch.randint(0, 256, (2, 64), device=device)  # batch=2, seq=64
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [2, 64, 256]")
    
    assert logits.shape == (2, 64, 256), f"Shape mismatch!"
    print("✓ Forward pass works!")
