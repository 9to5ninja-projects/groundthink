"""
GroundThink V4 Hybrid Model - Ratio Variants (GF-based)

Based on the winning GF (Gated Fusion) architecture.
These variants test different RWKV:Mamba balance ratios
by initializing the gate bias differently.

Variants:
- GF: Balanced (gate init ~0.5) - baseline winner
- GF_RH: RWKV-Heavy (gate init ~0.7, favors RWKV)
- GF_MH: Mamba-Heavy (gate init ~0.3, favors Mamba)

The gate learns during training, so this tests whether
starting with a bias toward one component helps.
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


class ParallelHybridBlock_GF_Ratio(nn.Module):
    """
    Parallel Hybrid Block with configurable gate bias.
    
    gate_init: Initial sigmoid(bias) value
    - 0.5 = balanced (default GF)
    - 0.7 = RWKV-heavy
    - 0.3 = Mamba-heavy
    
    fused = gate * rwkv + (1-gate) * mamba
    So higher gate = more RWKV contribution
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        layer_idx: int = 0,
        gate_init: float = 0.5,  # 0.5 = balanced, 0.7 = RWKV-heavy, 0.3 = Mamba-heavy
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
        
        # GF Fusion with configurable initial bias
        self.gate_proj = nn.Linear(hidden_size * 2, 1, bias=True)
        
        # Initialize bias to achieve target gate_init value
        # sigmoid(bias) = gate_init => bias = logit(gate_init)
        with torch.no_grad():
            # Clamp to avoid inf
            gate_init = max(0.01, min(0.99, gate_init))
            init_bias = math.log(gate_init / (1 - gate_init))  # logit function
            self.gate_proj.bias.fill_(init_bias)
            # Zero the weights so initial gate is purely from bias
            self.gate_proj.weight.zero_()
        
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
        
        # GF Fusion
        combined = torch.cat([out_rwkv, out_mamba], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        fused = gate * out_rwkv + (1 - gate) * out_mamba
        
        x = x + fused
        x = x + self.ffn(self.ffn_ln(x))
        
        if return_activations:
            return x, {'rwkv': out_rwkv, 'mamba': out_mamba, 'gate': gate.mean().item()}
        return x


class HybridModel_GF_Ratio(nn.Module):
    """Hybrid model with configurable gate bias for ratio testing"""
    
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
            ParallelHybridBlock_GF_Ratio(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                layer_idx=i,
                gate_init=gate_init,
            )
            for i in range(num_layers)
        ])
        
        self.ln_out = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        if tie_embeddings:
            self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Skip gate_proj - already initialized specially
                if hasattr(module, 'weight') and module.weight.shape[-1] != 257:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None and module.bias.numel() != 1:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, return_activations: bool = False):
        h = self.embed(x)
        
        all_activations = [] if return_activations else None
        
        for block in self.blocks:
            if return_activations:
                h, acts = block(h, return_activations=True)
                all_activations.append(acts)
            else:
                h = block(h)
        
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


# ============ Factory Functions ============

def create_hybrid_GF_RH_5m(vocab_size: int = 10000) -> HybridModel_GF_Ratio:
    """RWKV-Heavy: Gate biased toward RWKV (init 0.7)"""
    return HybridModel_GF_Ratio(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        ffn_mult=4.0,
        gate_init=0.7,  # Favor RWKV
    )


def create_hybrid_GF_MH_5m(vocab_size: int = 10000) -> HybridModel_GF_Ratio:
    """Mamba-Heavy: Gate biased toward Mamba (init 0.3)"""
    return HybridModel_GF_Ratio(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        ffn_mult=4.0,
        gate_init=0.3,  # Favor Mamba
    )


# Quick test
if __name__ == "__main__":
    print("Testing Ratio Variants...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, create_fn, expected_gate in [
        ("GF-RH (RWKV-Heavy)", create_hybrid_GF_RH_5m, 0.7),
        ("GF-MH (Mamba-Heavy)", create_hybrid_GF_MH_5m, 0.3),
    ]:
        print(f"\n{name}:")
        model = create_fn(vocab_size=256).to(device)
        
        x = torch.randint(0, 256, (2, 64), device=device)
        with torch.no_grad():
            logits, acts = model(x, return_activations=True)
        
        gate_values = [a['gate'] for a in acts]
        avg_gate = sum(gate_values) / len(gate_values)
        print(f"  Expected gate: {expected_gate}")
        print(f"  Actual avg gate: {avg_gate:.3f}")
        print(f"  Per-layer: {[f'{g:.3f}' for g in gate_values]}")
        
        del model
        torch.cuda.empty_cache()
    
    print("\n✓ Ratio variants work!")
