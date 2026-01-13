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

from ops import RWKV6Attention, Mamba2


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
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, 
                return_states: bool = False, rwkv_drop_prob: float = 0.0):
        """
        Forward pass through hybrid block.
        
        Args:
            x: Input tensor [B, T, hidden_size]
            return_activations: If True, return component outputs (Type A metrics)
            return_states: If True, return internal states (Type B metrics for S0-S4)
            rwkv_drop_prob: Probability of dropping RWKV during training
            
        Returns:
            If return_activations=False and return_states=False: output tensor
            If return_activations=True: (output, activations_dict)
            If return_states=True: (output, states_dict)
        """
        norm_x = self.ln(x)
        
        # PARALLEL pathways with optional state extraction
        if return_states:
            out_rwkv, _, rwkv_state_dict = self.rwkv6(norm_x, return_state=True)
            out_mamba, mamba_state_dict = self.mamba2(norm_x, return_state=True)
        else:
            out_rwkv, _, _ = self.rwkv6(norm_x)
            out_mamba = self.mamba2(norm_x)
        
        # RWKV Dropout: During training, randomly suppress RWKV to force Mamba learning
        if self.training and rwkv_drop_prob > 0.0:
            if torch.rand(1).item() < rwkv_drop_prob:
                out_rwkv = torch.zeros_like(out_rwkv)
        
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
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, 
                return_states: bool = False, rwkv_drop_prob: float = 0.0):
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input token IDs [B, T]
            return_activations: If True, return component outputs (Type A metrics)
            return_states: If True, return internal states (Type B metrics for S0-S4)
            rwkv_drop_prob: Probability of dropping RWKV during training
            
        Returns:
            If return_activations=False and return_states=False: logits tensor
            If return_activations=True: (logits, list of activation dicts per layer)
            If return_states=True: (logits, states_dict) with aggregated states from last layer
        """
        h = self.embed(x)
        
        all_activations = [] if return_activations else None
        all_states = [] if return_states else None
        
        for block in self.blocks:
            if return_states:
                h, states = block(h, return_states=True, rwkv_drop_prob=rwkv_drop_prob)
                all_states.append(states)
            elif return_activations:
                h, acts = block(h, return_activations=True, rwkv_drop_prob=rwkv_drop_prob)
                all_activations.append(acts)
            else:
                h = block(h, rwkv_drop_prob=rwkv_drop_prob)
        
        h = self.ln_out(h)
        logits = self.head(h)
        
        if return_states:
            # Aggregate states from last layer (S0-S4 tests use this)
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


def create_hybrid_GF_XM_5m(vocab_size: int = 10000) -> HybridModel_GF_Ratio:
    """eXtreme Mamba: Gate heavily biased toward Mamba (init 0.03)
    
    Hypothesis: If optimizer drifts from 70/30 to 90/10 (RWKV dominant),
    starting at 97/3 (Mamba) might drift to ~70/30 — actually balanced.
    """
    return HybridModel_GF_Ratio(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        ffn_mult=4.0,
        gate_init=0.03,  # 3% RWKV, 97% Mamba at init
    )


def create_hybrid_GF_XR_5m(vocab_size: int = 10000) -> HybridModel_GF_Ratio:
    """eXtreme RWKV: Gate heavily biased toward RWKV (init 0.97)
    
    Control experiment: Start nearly-pure RWKV to see drift direction.
    """
    return HybridModel_GF_Ratio(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        ffn_mult=4.0,
        gate_init=0.97,  # 97% RWKV, 3% Mamba at init
    )


# Quick test
if __name__ == "__main__":
    print("Testing Ratio Variants...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, create_fn, expected_gate in [
        ("GF-RH (RWKV-Heavy)", create_hybrid_GF_RH_5m, 0.7),
        ("GF-MH (Mamba-Heavy)", create_hybrid_GF_MH_5m, 0.3),
        ("GF-XM (eXtreme Mamba)", create_hybrid_GF_XM_5m, 0.03),
        ("GF-XR (eXtreme RWKV)", create_hybrid_GF_XR_5m, 0.97),
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


# ============ Phase 1: GRU Arbiter Variants (Task 0.1) ============

from ops.arbiter_gru import GRUArbiter


class ParallelHybridBlock_GRU(nn.Module):
    """
    Parallel Hybrid Block with GRU Arbiter fusion (Phase 1).
    
    Replaces scalar gate with stateful GRU that learns to weight
    RWKV (amplifier) vs Mamba (damper) based on sequence context.
    
    Phase 0 findings inform design:
    - RWKV amplifies 1.28x per layer
    - Mamba damps at layer level (0.005x)
    - Target total variance: 2-6x
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        layer_idx: int = 0,
        dropout: float = 0.0,
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
        
        # GRU Arbiter fusion (Phase 1 upgrade)
        self.fusion = GRUArbiter(hidden_size, dropout=dropout)
        
        # Store last fusion weights for loss computation (Task 0.3)
        self._last_fusion_weights = None
        
        # FFN
        ffn_hidden = int(hidden_size * ffn_mult)
        self.ffn_ln = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size, bias=False),
        )
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, 
                return_states: bool = False, rwkv_drop_prob: float = 0.0,
                arbiter_hidden: torch.Tensor = None):
        """
        Forward pass through hybrid block with GRU arbiter.
        
        Args:
            x: Input tensor [B, T, hidden_size]
            return_activations: If True, return component outputs
            return_states: If True, return internal states
            rwkv_drop_prob: Probability of dropping RWKV during training
            arbiter_hidden: Previous GRU hidden state (for cross-chunk statefulness)
            
        Returns:
            If return_activations=False and return_states=False: output tensor
            If return_activations=True: (output, activations_dict)
            If return_states=True: (output, states_dict)
        """
        norm_x = self.ln(x)
        
        # PARALLEL pathways
        if return_states:
            out_rwkv, _, rwkv_state_dict = self.rwkv6(norm_x, return_state=True)
            out_mamba, mamba_state_dict = self.mamba2(norm_x, return_state=True)
        else:
            out_rwkv, _, _ = self.rwkv6(norm_x)
            out_mamba = self.mamba2(norm_x)
        
        # RWKV Dropout: During training, randomly suppress RWKV to force Mamba learning
        if self.training and rwkv_drop_prob > 0.0:
            if torch.rand(1).item() < rwkv_drop_prob:
                out_rwkv = torch.zeros_like(out_rwkv)
        
        # GRU Arbiter Fusion (Phase 1)
        fused, fusion_weights, new_hidden = self.fusion(out_rwkv, out_mamba, hidden=arbiter_hidden)
        
        # Store for Task 0.3 loss computation
        self._last_fusion_weights = fusion_weights  # (B, L, 2)
        
        x = x + fused
        x = x + self.ffn(self.ffn_ln(x))
        
        if return_states:
            # Compute mean weights for logging (comparable to old scalar gate)
            w_rwkv_mean = fusion_weights[:, :, 0].mean().item()
            w_mamba_mean = fusion_weights[:, :, 1].mean().item()
            
            states = {
                'rwkv_state': rwkv_state_dict.get('rwkv_state') if rwkv_state_dict else None,
                'mamba_state': mamba_state_dict.get('mamba_state') if mamba_state_dict else None,
                'fusion_weights': fusion_weights,  # Full tensor for analysis
                'w_rwkv': w_rwkv_mean,
                'w_mamba': w_mamba_mean,
                'arbiter_hidden': new_hidden,
            }
            return x, states
        
        if return_activations:
            w_rwkv_mean = fusion_weights[:, :, 0].mean().item()
            return x, {
                'rwkv': out_rwkv, 
                'mamba': out_mamba, 
                'gate': w_rwkv_mean,  # Backward compatible key
                'fusion_weights': fusion_weights,
            }
        return x


class HybridModel_GRU(nn.Module):
    """Hybrid model with GRU Arbiter fusion (Phase 1).
    
    Replaces static gated fusion with stateful GRU arbiter.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 128,
        num_layers: int = 8,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        tie_embeddings: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        self.blocks = nn.ModuleList([
            ParallelHybridBlock_GRU(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                layer_idx=i,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        
        self.ln_out = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        if tie_embeddings:
            self.head.weight = self.embed.weight
        
        # BlinkDL initialization (Phase 0 mandate)
        self._init_weights()
    
    def _init_weights(self):
        """Apply BlinkDL initialization for training stability."""
        # Embedding: small uniform
        nn.init.uniform_(self.embed.weight, -1e-4, 1e-4)
        
        # Head: zero if not tied (tied inherits from embed)
        if self.head.weight is not self.embed.weight:
            nn.init.zeros_(self.head.weight)
    
    def forward(self, x: torch.Tensor, return_activations: bool = False, 
                return_states: bool = False, rwkv_drop_prob: float = 0.0):
        """
        Forward pass through full model.
        
        Args:
            x: Input token indices [B, T]
            return_activations: If True, return per-layer activations
            return_states: If True, return per-layer states
            rwkv_drop_prob: Probability of dropping RWKV during training
        """
        h = self.embed(x)
        
        all_activations = [] if return_activations else None
        all_states = [] if return_states else None
        
        for block in self.blocks:
            if return_states:
                h, states = block(h, return_states=True, rwkv_drop_prob=rwkv_drop_prob)
                all_states.append(states)
            elif return_activations:
                h, acts = block(h, return_activations=True, rwkv_drop_prob=rwkv_drop_prob)
                all_activations.append(acts)
            else:
                h = block(h, rwkv_drop_prob=rwkv_drop_prob)
        
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
    
    def get_fusion_weights(self):
        """Collect fusion weights from all layers for loss computation."""
        weights = []
        for block in self.blocks:
            if block._last_fusion_weights is not None:
                weights.append(block._last_fusion_weights)
        return weights


def create_hybrid_GRU_5m(vocab_size: int = 10000) -> HybridModel_GRU:
    """Phase 1 GRU Arbiter model at ~5M params."""
    return HybridModel_GRU(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=8,
        num_heads=4,
        ffn_mult=4.0,
        dropout=0.0,
    )
