"""
GroundedMamba Model
Per FOUNDATION.md specification

Start config: vocab_size=10000, dim=512, depth=8, ssm_dim=16, rwkv_heads=4
"""

import torch
import torch.nn as nn
from typing import Optional

from grounded_layers import GroundedMambaBlock


class GroundedMamba(nn.Module):
    """
    Full GroundedMamba model.
    
    Combines:
    - StableStateSSM for dynamic "thinking"
    - RWKVGroundedMemory for stable "grounding"
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        dim: int = 512,
        depth: int = 8,
        ssm_dim: int = 16,
        rwkv_heads: int = 4,
        fixed_gate: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Blocks
        self.blocks = nn.ModuleList([
            GroundedMambaBlock(
                dim=dim,
                ssm_dim=ssm_dim,
                num_heads=rwkv_heads,
                fixed_gate=fixed_gate,
            )
            for _ in range(depth)
        ])
        
        # Final norm
        self.ln_out = nn.LayerNorm(dim)
        
        # Output projection
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        
    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[list] = None,
        return_states: bool = True,
    ):
        """
        Args:
            input_ids: (B, L) token indices
            states: list of state dicts per layer, or None
            return_states: whether to return updated states
            
        Returns:
            logits: (B, L, vocab_size)
            states: list of state dicts if return_states=True
        """
        B, L = input_ids.shape
        
        # Initialize states if needed
        if states is None:
            states = [None] * self.depth
        
        # Embed
        x = self.embedding(input_ids)
        
        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            layer_state = states[i]
            ssm_state = layer_state['ssm'] if layer_state else None
            wkv_state = layer_state['wkv'] if layer_state else None
            
            x, state_dict = block(x, ssm_state, wkv_state)
            new_states.append(state_dict)
        
        # Final norm and projection
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        if return_states:
            return logits, new_states
        return logits, None
    
    def reset_states(self):
        """Reset all internal states"""
        # States are passed explicitly, nothing internal to reset
        pass
    
    def get_param_groups(self, lr: float):
        """
        Get parameter groups with different LRs per FOUNDATION.md:
        - SSM params: lr * 0.5, weight_decay=0.1
        - RWKV params: lr * 0.3, weight_decay=0.01
        - Norm params: lr * 1.0, weight_decay=0.0
        - Other params: lr * 1.0, weight_decay=0.1
        """
        ssm_params = []
        rwkv_params = []
        norm_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'ssm' in name.lower():
                ssm_params.append(param)
            elif 'rwkv' in name.lower() or 'memory' in name.lower() or 'time_decay' in name or 'time_first' in name:
                rwkv_params.append(param)
            elif 'norm' in name.lower() or 'ln' in name.lower():
                norm_params.append(param)
            else:
                other_params.append(param)
        
        return [
            {'params': ssm_params, 'lr': lr * 0.5, 'weight_decay': 0.1},
            {'params': rwkv_params, 'lr': lr * 0.3, 'weight_decay': 0.01},
            {'params': norm_params, 'lr': lr * 1.0, 'weight_decay': 0.0},
            {'params': other_params, 'lr': lr * 1.0, 'weight_decay': 0.1},
        ]
