"""
GroundThink V4 Hybrid Model (GF - Gated Fusion)

TRUE PARALLEL ARCHITECTURE: RWKV-6 and Mamba-2 run side-by-side in EVERY block.
Both kernels process the same normalized input, outputs combined via learned gate.

Fusion Type: GF (Gated Fusion)
- Concatenate outputs, project to gate value
- gate = sigmoid(Linear([rwkv, mamba]))
- fused = gate * rwkv + (1-gate) * mamba
- 257 fusion params at hidden=128 (256+1 bias)

Goal: Break the 7.0 Loss Wall through gradient independence of parallel pathways.
"""

# Fix OpenMP conflict BEFORE any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# FLA library imports
from fla_replacements import RWKV6Attention
from fla_replacements import Mamba2


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


class ParallelHybridBlock_GF(nn.Module):
    """
    Parallel Hybrid Block: RWKV-6 and Mamba-2 run IN PARALLEL
    
    Fusion Type: GF (Gated Fusion)
    - Concat outputs -> project to scalar gate
    - gate = sigmoid(projection)
    - fused = gate * rwkv + (1-gate) * mamba
    - 257 fusion params at hidden=128
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
        self.rwkv6 = RWKV6Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_idx=layer_idx,
        )
        
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
        
        # GF Fusion: Gated Fusion (257 params at hidden=128)
        # Input: [batch, seq, hidden*2] -> Output: [batch, seq, 1]
        self.gate_proj = nn.Linear(hidden_size * 2, 1, bias=True)
        
        # FFN with 4x expansion
        ffn_hidden = int(hidden_size * ffn_mult)
        self.ffn_ln = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size, bias=False),
        )
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, rwkv_drop_prob: float = 0.0):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            return_activations: If True, also return component outputs for monitoring
            rwkv_drop_prob: Probability of dropping RWKV output entirely (training only)
                           Forces Mamba to learn independently when RWKV is suppressed
        Returns:
            out: [batch_size, seq_len, hidden_size]
            activations (optional): dict with 'rwkv', 'mamba', and 'gate' tensors
        """
        # Pre-norm
        norm_x = self.ln(x)
        
        # PARALLEL pathways
        out_rwkv, _, _ = self.rwkv6(norm_x)
        out_mamba = self.mamba2(norm_x)
        
        # RWKV Dropout: During training, randomly suppress RWKV to force Mamba learning
        if self.training and rwkv_drop_prob > 0.0:
            if torch.rand(1).item() < rwkv_drop_prob:
                out_rwkv = torch.zeros_like(out_rwkv)
        
        # GF Fusion: Gated Fusion
        combined = torch.cat([out_rwkv, out_mamba], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))  # [batch, seq, 1]
        fused = gate * out_rwkv + (1 - gate) * out_mamba
        
        # Residual connection
        x = x + fused
        
        # FFN with residual
        x = x + self.ffn(self.ffn_ln(x))
        
        if return_activations:
            return x, {'rwkv': out_rwkv, 'mamba': out_mamba, 'gate': gate.mean().item()}
        return x


class HybridModel_GF(nn.Module):
    """
    GroundThink V4 Hybrid Language Model (GF Fusion)
    
    Stack of ParallelHybridBlock_GF with shared embeddings.
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
            ParallelHybridBlock_GF(
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
        
        # Tie embeddings
        if tie_embeddings:
            self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, rwkv_drop_prob: float = 0.0):
        """
        Args:
            x: Input token IDs [batch_size, seq_len]
            return_activations: If True, also return component activations
            rwkv_drop_prob: Probability of dropping RWKV output (anneals during training)
        """
        h = self.embed(x)
        
        all_activations = [] if return_activations else None
        
        for block in self.blocks:
            if return_activations:
                h, acts = block(h, return_activations=True, rwkv_drop_prob=rwkv_drop_prob)
                all_activations.append(acts)
            else:
                h = block(h, rwkv_drop_prob=rwkv_drop_prob)
        
        h = self.ln_out(h)
        logits = self.head(h)
        
        if return_activations:
            return logits, all_activations
        return logits
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
        return n_params


def create_hybrid_GF_5m(vocab_size: int = 10000) -> HybridModel_GF:
    """Create ~5M parameter hybrid model with GF fusion"""
    return HybridModel_GF(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        ffn_mult=4.0,
    )


# Quick test
if __name__ == "__main__":
    print("Testing HybridModel_GF (Gated Fusion)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_hybrid_GF_5m(vocab_size=256).to(device)
    print(f"Model created: {model.get_num_params():,} non-embedding params")
    
    # Count fusion params (gate_proj per layer)
    fusion_params = sum(p.numel() for n, p in model.named_parameters() if 'gate_proj' in n)
    print(f"Fusion params: {fusion_params:,} (257 per layer)")
    
    # Test forward pass
    x = torch.randint(0, 256, (2, 64), device=device)
    
    with torch.no_grad():
        logits, acts = model(x, return_activations=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Show gate values
    gate_values = [a['gate'] for a in acts]
    print(f"Gate values per layer: {[f'{g:.3f}' for g in gate_values]}")
    
    assert logits.shape == (2, 64, 256), f"Shape mismatch!"
    print("✓ Forward pass works!")
