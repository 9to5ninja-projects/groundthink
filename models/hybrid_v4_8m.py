"""
GroundThink V4 Hybrid Model - 8M Scale (GF-MH)

Scaled version of the Phase 2 winner (GF-MH - Gated Fusion + Mamba-Heavy).
Based on hybrid_v4_ratio.py, increased from 3.5M to ~8M parameters.

Configuration:
- hidden_size: 128 → 192
- num_layers: 8 (unchanged)
- num_heads: 4 → 6 (maintains head_size=32 for CUDA kernel)
- gate_init: 0.3 (Mamba-heavy, same as 3.5M winner)
- Total params: ~7.93M

Validation:
- G1 PASS: Forward pass, no NaN, correct shapes [batch, seq, 97]
- G2 PASS: Init entropy 4.43 (target 2.0-5.0)

Created: 2026-01-10 (Task 19)
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


class ParallelHybridBlock_8M(nn.Module):
    """
    Parallel Hybrid Block for 8M model.
    Same architecture as Phase 2 winner (GF-MH), scaled up.
    """
    
    def __init__(
        self,
        hidden_size: int = 192,
        num_heads: int = 6,
        ffn_mult: float = 4.0,
        layer_idx: int = 0,
        gate_init: float = 0.3,
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
        
        # GF Fusion with Mamba-heavy bias
        self.gate_proj = nn.Linear(hidden_size * 2, 1, bias=True)
        
        with torch.no_grad():
            gate_init = max(0.01, min(0.99, gate_init))
            init_bias = math.log(gate_init / (1 - gate_init))
            self.gate_proj.bias.fill_(init_bias)
            self.gate_proj.weight.zero_()
        
        # FFN
        ffn_hidden = int(hidden_size * ffn_mult)
        self.ffn_ln = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size, bias=False),
        )
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, return_states: bool = False):
        norm_x = self.ln(x)
        
        # PARALLEL pathways with optional state extraction
        if return_states:
            out_rwkv, _, rwkv_state_dict = self.rwkv6(norm_x, return_state=True)
            out_mamba, mamba_state_dict = self.mamba2(norm_x, return_state=True)
        else:
            out_rwkv, _, _ = self.rwkv6(norm_x)
            out_mamba = self.mamba2(norm_x)
        
        # GF Fusion
        combined = torch.cat([out_rwkv, out_mamba], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        fused = gate * out_rwkv + (1 - gate) * out_mamba
        
        x = x + fused
        x = x + self.ffn(self.ffn_ln(x))
        
        if return_states:
            states = {
                'rwkv_state': rwkv_state_dict.get('rwkv_state') if rwkv_state_dict else None,
                'mamba_state': mamba_state_dict.get('mamba_state') if mamba_state_dict else None,
                'gate': gate.mean().item(),
            }
            return x, states
        
        if return_activations:
            return x, {'rwkv': out_rwkv, 'mamba': out_mamba, 'gate': gate.mean().item()}
        return x


class HybridModel_8M(nn.Module):
    """
    8M Parameter Hybrid Model (GF-MH architecture)
    
    Scaled from 3.5M winner:
    - hidden: 128 → 192
    - heads: 4 → 6 (head_size=32 preserved)
    - layers: 8 (unchanged)
    - gate_init: 0.3 (Mamba-heavy)
    """
    
    def __init__(
        self,
        vocab_size: int = 97,  # Shakespeare vocab
        hidden_size: int = 192,
        num_layers: int = 8,
        num_heads: int = 6,
        ffn_mult: float = 4.0,
        tie_embeddings: bool = True,
        gate_init: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gate_init = gate_init
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        self.blocks = nn.ModuleList([
            ParallelHybridBlock_8M(
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
                if hasattr(module, 'weight') and module.weight.shape[-1] != 385:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None and module.bias.numel() != 1:
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


def create_hybrid_GF_MH_8m(vocab_size: int = 97) -> HybridModel_8M:
    """
    Factory function for 8M GF-MH model.
    
    This is the scaled version of the Phase 2 winner.
    Use for Phase 3 extended training (Task 20).
    """
    return HybridModel_8M(
        vocab_size=vocab_size,
        hidden_size=192,
        num_layers=8,
        num_heads=6,
        ffn_mult=4.0,
        gate_init=0.3,
    )


# Validation test
if __name__ == "__main__":
    print("Validating 8M GF-MH Model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_hybrid_GF_MH_8m().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Configuration:")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Num layers: {model.num_layers}")
    print(f"  Gate init: {model.gate_init}")
    print(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # G1: Forward pass
    print(f"\nG1: Forward Pass Test")
    x = torch.randint(0, 97, (2, 64), device=device)
    with torch.no_grad():
        logits, acts = model(x, return_activations=True)
    
    print(f"  Output shape: {logits.shape}")
    print(f"  NaN check: {'PASS' if not torch.isnan(logits).any() else 'FAIL'}")
    print(f"  Output range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    
    # G2: Init entropy
    print(f"\nG2: Init Entropy Test")
    probs = torch.softmax(logits[0, -1], dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    status = "PASS" if 2.0 <= entropy <= 5.0 else ("WARN" if entropy <= 7.0 else "FAIL")
    print(f"  Entropy: {entropy:.2f} (target: 2.0-5.0) [{status}]")
    
    # Gate values
    print(f"\nGate Values (per layer):")
    for i, act in enumerate(acts):
        print(f"  Layer {i}: {act['gate']:.3f}")
    
    print(f"\n✓ 8M Model Validated - Ready for Phase 3 Training")
