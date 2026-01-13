"""GRU Arbiter for Twin Debate fusion.

Task 0.1 implementation: Stateful fusion arbiter that learns to weight
RWKV (amplifier) vs Mamba (damper) contributions based on sequence context.

Phase 0 findings inform design:
- RWKV amplifies variance per-layer (1.28x)
- Mamba damps at layer level (0.005x)
- Target total variance: 2-6x (SSM range)
- BlinkDL zero-init on projections is mandatory
"""

import torch
import torch.nn as nn


class GRUArbiter(nn.Module):
    """Stateful fusion arbiter for Twin Debate architecture.
    
    Learns to weight RWKV (amplifier) vs Mamba (damper) contributions
    based on sequence context maintained in GRU hidden state.
    
    Args:
        d_model: Hidden dimension of expert outputs
        dropout: Dropout on GRU hidden state (default: 0.0)
    """
    
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        
        # GRU cell for stateful decision making
        # Input: concat([rwkv_out, mamba_out]) = 2*d_model
        # Hidden: d_model (compressed context)
        self.gru = nn.GRUCell(
            input_size=2 * d_model,
            hidden_size=d_model
        )
        
        # Project hidden state to fusion weights [w_rwkv, w_mamba]
        self.to_weights = nn.Linear(d_model, 2)
        
        # Optional: dropout on GRU output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize with BlinkDL-style zeros
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Phase 0 mandate: zero-init projections to preserve variance."""
        nn.init.zeros_(self.to_weights.weight)
        nn.init.zeros_(self.to_weights.bias)
        # GRU internal params use PyTorch defaults (proven stable)
    
    def forward(
        self,
        rwkv_out: torch.Tensor,  # (B, L, D)
        mamba_out: torch.Tensor,  # (B, L, D)
        hidden: torch.Tensor = None  # (B, D) or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rwkv_out: RWKV expert output
            mamba_out: Mamba expert output (post-residual in Task 0.2)
            hidden: Previous GRU hidden state (None for sequence start)
        
        Returns:
            fused: Weighted combination (B, L, D)
            weights: Fusion weights (B, L, 2) [w_rwkv, w_mamba]
            hidden: Updated GRU state (B, D)
        """
        B, L, D = rwkv_out.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(B, D, device=rwkv_out.device, dtype=rwkv_out.dtype)
        
        outputs = []
        weight_history = []
        
        # Process sequence step-by-step to maintain GRU state
        for t in range(L):
            # Concat expert outputs as GRU input
            gru_input = torch.cat([rwkv_out[:, t], mamba_out[:, t]], dim=-1)  # (B, 2D)
            
            # Update hidden state
            hidden = self.gru(gru_input, hidden)  # (B, D)
            hidden = self.dropout(hidden)
            
            # Compute fusion weights from hidden state
            logits = self.to_weights(hidden)  # (B, 2)
            weights = torch.softmax(logits, dim=-1)  # (B, 2) - sum to 1
            
            # Fuse expert outputs
            w_rwkv = weights[:, 0:1]  # (B, 1)
            w_mamba = weights[:, 1:2]  # (B, 1)
            
            fused_t = w_rwkv * rwkv_out[:, t] + w_mamba * mamba_out[:, t]  # (B, D)
            
            outputs.append(fused_t)
            weight_history.append(weights)
        
        # Stack time dimension
        fused = torch.stack(outputs, dim=1)  # (B, L, D)
        weights = torch.stack(weight_history, dim=1)  # (B, L, 2)
        
        return fused, weights, hidden
