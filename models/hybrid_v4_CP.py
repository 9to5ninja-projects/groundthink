"""
GroundThink V4 Hybrid Model (CP - Concatenate + Project Fusion)

TRUE PARALLEL ARCHITECTURE: RWKV-6 and Mamba-2 run side-by-side in EVERY block.
Both kernels process the same normalized input, outputs combined via projection.

Fusion Type: CP (Concatenate + Project)
- Concatenate RWKV and Mamba outputs along hidden dim
- Project back to hidden_size via Linear layer
- ~33K fusion params at hidden=128 (128*2 -> 128)

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
from cuda_backends import RWKV6Attention
from cuda_backends import Mamba2


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


class ParallelHybridBlock_CP(nn.Module):
    """
    Parallel Hybrid Block: RWKV-6 and Mamba-2 run IN PARALLEL
    
    Fusion Type: CP (Concatenate + Project)
    - Concat [rwkv_out, mamba_out] -> [batch, seq, hidden*2]
    - Linear projection back to hidden_size
    - 33K fusion params at hidden=128
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
        
        # CP Fusion: Concatenate + Project (33K params at hidden=128)
        self.fusion_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # FFN with 4x expansion
        ffn_hidden = int(hidden_size * ffn_mult)
        self.ffn_ln = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size, bias=False),
        )
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, return_states: bool = False):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            return_activations: If True, also return component outputs for monitoring
            return_states: If True, return internal states for S0-S4 tests
        """
        # Pre-norm
        norm_x = self.ln(x)
        
        # PARALLEL pathways with optional state extraction
        if return_states:
            out_rwkv, _, rwkv_state_dict = self.rwkv6(norm_x, return_state=True)
            out_mamba, mamba_state_dict = self.mamba2(norm_x, return_state=True)
        else:
            out_rwkv, _, _ = self.rwkv6(norm_x)
            out_mamba = self.mamba2(norm_x)
        
        # CP Fusion: Concatenate + Project
        combined = torch.cat([out_rwkv, out_mamba], dim=-1)
        fused = self.fusion_proj(combined)
        
        # Residual connection
        x = x + fused
        
        # FFN with residual
        x = x + self.ffn(self.ffn_ln(x))
        
        if return_states:
            states = {
                'rwkv_state': rwkv_state_dict.get('rwkv_state') if rwkv_state_dict else None,
                'mamba_state': mamba_state_dict.get('mamba_state') if mamba_state_dict else None,
                'gate': 0.5,  # CP uses learned projection, no explicit gate
            }
            return x, states
        
        if return_activations:
            return x, {'rwkv': out_rwkv, 'mamba': out_mamba}
        return x


class HybridModel_CP(nn.Module):
    """
    GroundThink V4 Hybrid Language Model (CP Fusion)
    
    Stack of ParallelHybridBlock_CP with shared embeddings.
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
            ParallelHybridBlock_CP(
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
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, return_states: bool = False):
        h = self.embed(x)
        
        all_activations = [] if return_activations else None
        all_states = [] if return_states else None
        
        for block in self.blocks:
            if return_states:
                h, states = block(h, return_states=True)
                all_states.append(states)
            elif return_activations:
                h, acts = block(h, return_activations=True)
                all_activations.append(acts)
            else:
                h = block(h)
        
        h = self.ln_out(h)
        logits = self.head(h)
        
        if return_states:
            last_states = all_states[-1] if all_states else {}
            return logits, last_states
        
        if return_activations:
            return logits, all_activations
        return logits
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
        return n_params


def create_hybrid_CP_5m(vocab_size: int = 10000) -> HybridModel_CP:
    """Create ~5M parameter hybrid model with CP fusion"""
    return HybridModel_CP(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        ffn_mult=4.0,
    )


# Quick test
if __name__ == "__main__":
    print("Testing HybridModel_CP (Concatenate + Project)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_hybrid_CP_5m(vocab_size=256).to(device)
    print(f"Model created: {model.get_num_params():,} non-embedding params")
    
    # Count fusion params
    fusion_params = sum(p.numel() for n, p in model.named_parameters() if 'fusion' in n)
    print(f"Fusion params: {fusion_params:,}")
    
    # Test forward pass
    x = torch.randint(0, 256, (2, 64), device=device)
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 64, 256), f"Shape mismatch!"
    print("✓ Forward pass works!")
