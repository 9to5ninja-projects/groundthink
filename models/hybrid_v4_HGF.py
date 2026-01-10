"""
GroundThink V4 Hybrid Model (HGF - Hybrid-Gated Fusion)

Per-position AND per-dimension gating.
Combines the best of HY (dimension control) and GF (position adaptivity).

Fusion Type: HGF (Hybrid-Gated Fusion)
- gate = sigmoid(Linear([rwkv, mamba])) → [batch, seq, hidden]
- fused = gate * rwkv + (1-gate) * mamba (elementwise)
- ~33K fusion params at hidden=128 (same as CP)

Key difference from CP:
- CP: Arbitrary linear combination (can mix dims freely)
- HGF: Constrained to interpolation (each dim is blend of same dim)

This gives explicit control over the "shape" of the learning field.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla_replacements import RWKV6Attention, Mamba2


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


class ParallelHybridBlock_HGF(nn.Module):
    """
    Parallel Hybrid Block with per-position, per-dimension gating.
    
    gate: [batch, seq, hidden] — different blend for each position AND dimension
    fused = gate * rwkv + (1-gate) * mamba
    
    This is the "maximum control" variant.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        layer_idx: int = 0,
        gate_init: float = 0.5,  # Initial blend: 0.5 = balanced
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        
        # Pre-norm
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
        
        # HGF Fusion: Per-position, per-dimension gates
        # Input: [batch, seq, hidden*2] → Output: [batch, seq, hidden]
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        
        # Initialize for target gate value
        with torch.no_grad():
            gate_init = max(0.01, min(0.99, gate_init))
            init_bias = math.log(gate_init / (1 - gate_init))
            self.gate_proj.bias.fill_(init_bias)
            # Small random init for weights (not zero — we want per-dim learning)
            nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.1)
        
        # FFN
        ffn_hidden = int(hidden_size * ffn_mult)
        self.ffn_ln = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size, bias=False),
        )
    
    def forward(self, x: torch.Tensor, return_activations: bool = False):
        norm_x = self.ln(x)
        
        # PARALLEL pathways
        out_rwkv, _, _ = self.rwkv6(norm_x)
        out_mamba = self.mamba2(norm_x)
        
        # HGF Fusion: per-position, per-dimension
        combined = torch.cat([out_rwkv, out_mamba], dim=-1)  # [B, S, 256]
        gate = torch.sigmoid(self.gate_proj(combined))        # [B, S, 128]
        fused = gate * out_rwkv + (1 - gate) * out_mamba      # Elementwise
        
        x = x + fused
        x = x + self.ffn(self.ffn_ln(x))
        
        if return_activations:
            return x, {
                'rwkv': out_rwkv,
                'mamba': out_mamba,
                'gate_mean': gate.mean().item(),
                'gate_std': gate.std().item(),
            }
        return x


class HybridModel_HGF(nn.Module):
    """
    GroundThink V4 Hybrid Language Model (HGF Fusion)
    
    Per-position, per-dimension gating for maximum control.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 128,
        num_layers: int = 8,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        tie_embeddings: bool = True,
        gate_init: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gate_init = gate_init
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        self.blocks = nn.ModuleList([
            ParallelHybridBlock_HGF(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                layer_idx=i,
                gate_init=gate_init,
            )
            for i in range(num_layers)
        ])
        
        self.ln_out = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if not hasattr(module, '_no_init'):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, return_activations: bool = False):
        h = self.embed(x)
        
        all_activations = []
        for block in self.blocks:
            if return_activations:
                h, acts = block(h, return_activations=True)
                all_activations.append(acts)
            else:
                h = block(h)
        
        h = self.ln_out(h)
        logits = self.lm_head(h)
        
        if return_activations:
            return logits, all_activations
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Convenience aliases
def create_hgf_balanced(vocab_size=97, **kwargs):
    """HGF with balanced init (50/50)"""
    return HybridModel_HGF(vocab_size=vocab_size, gate_init=0.5, **kwargs)


def create_hgf_mamba_heavy(vocab_size=97, **kwargs):
    """HGF with Mamba-heavy init (30% RWKV, 70% Mamba)"""
    return HybridModel_HGF(vocab_size=vocab_size, gate_init=0.3, **kwargs)


def create_hgf_rwkv_heavy(vocab_size=97, **kwargs):
    """HGF with RWKV-heavy init (70% RWKV, 30% Mamba)"""
    return HybridModel_HGF(vocab_size=vocab_size, gate_init=0.7, **kwargs)
