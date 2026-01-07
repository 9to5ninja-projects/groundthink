"""
GroundThink Core Layers

Key insight: RWKV-7 already has input-dependent selective decay.
The innovation here is the GroundingMechanism that ensures state stability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (norm + self.eps) * self.weight


class GroundingMechanism(nn.Module):
    """
    Ensures state doesn't drift - the key innovation over vanilla RWKV-7.
    
    Three-tier grounding:
    1. Local (conv) - captures n-grams before recurrence
    2. Medium-term (learned base decay) - provides stability floor
    3. Long-term (residual pathway) - ensures gradient flow
    """
    
    def __init__(self, dim: int, kernel_size: int = 4, min_retention: float = 0.01, max_retention: float = 0.99):
        super().__init__()
        self.dim = dim
        self.min_retention = min_retention
        self.max_retention = max_retention
        
        # Tier 1: Local grounding via depthwise conv
        self.conv_short = nn.Conv1d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size - 1,
            groups=dim,
            bias=False
        )
        
        # Tier 2: Learned base decay (stability floor)
        self.base_decay = nn.Parameter(torch.ones(dim))
        
        # Tier 3: Residual weight for long-term grounding
        self.res_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, selective_decay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, L, D)
            selective_decay: Input-dependent decay from RWKV-7 (B, L, D)
        
        Returns:
            grounded_x: Input with local context
            grounded_decay: Combined decay factor (stable + selective)
        """
        B, L, D = x.shape
        
        # Tier 1: Local n-gram grounding
        x_conv = self.conv_short(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        grounded_x = x + 0.1 * x_conv  # Small residual from local context
        
        # Tier 2: Combine base decay with selective decay
        # Base decay provides stability floor
        w_base = torch.exp(-self.base_decay.view(1, 1, -1))
        
        # Selective decay from RWKV-7's input-dependent mechanism
        w_selective = selective_decay  # Already exp(-w) form
        
        # Combined: grounding + thinking
        w_combined = w_base * w_selective
        
        # Tier 3: Clamp to ensure minimum retention (never completely forget)
        # and maximum decay (always some forgetting to prevent saturation)
        grounded_decay = torch.clamp(w_combined, min=self.min_retention, max=self.max_retention)
        
        return grounded_x, grounded_decay


class StateNormalizer(nn.Module):
    """
    Prevents state explosion in long sequences.
    RMS normalization applied to state matrices.
    """
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state to prevent explosion"""
        norm = state.pow(2).mean(-1, keepdim=True).sqrt()
        return state / (norm + self.eps)


class TimeMixing(nn.Module):
    """
    RWKV-7 Time Mixing with Grounding.
    
    Uses the existing RWKV-7 selective mechanism but adds grounding
    to prevent state drift in very long contexts.
    """
    
    def __init__(
        self, 
        dim: int,
        n_heads: int,
        head_dim: int = 64,
        state_dim: int = 64,
        use_grounding: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.state_dim = state_dim
        self.use_grounding = use_grounding
        
        # RWKV-7 projections
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        
        # Time decay (w) - made input-dependent in RWKV-7
        self.time_decay = nn.Linear(dim, dim, bias=False)
        
        # Selection parameters (a, kk in RWKV-7 notation)
        self.alpha = nn.Linear(dim, dim, bias=False)  # a
        self.beta = nn.Linear(dim, dim, bias=False)   # kk
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Grounding mechanism
        if use_grounding:
            self.grounding = GroundingMechanism(dim)
        
        # State normalizer
        self.state_norm = StateNormalizer()
        
        # Normalization
        self.ln = RMSNorm(dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for stability"""
        # Zero-init output for residual start (O3 initialization)
        nn.init.zeros_(self.out_proj.weight)
        
        # Small positive gate init
        nn.init.uniform_(self.gate.weight, 0.9, 1.1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input (B, L, D)
            state: Previous hidden state (B, H, state_dim, head_dim) or None
        
        Returns:
            output: (B, L, D)
            new_state: (B, H, state_dim, head_dim)
        """
        B, L, D = x.shape
        H = self.n_heads
        
        # Project inputs
        r = torch.sigmoid(self.receptance(x))  # Receptance (read gate)
        k = self.key(x)
        v = self.value(x)
        g = torch.sigmoid(self.gate(x))  # Output gate
        
        # Input-dependent decay (the "selective" part, already in RWKV-7)
        w = self.time_decay(x)
        w = torch.exp(-torch.exp(w))  # Double exp for numerical stability
        
        # Selection parameters
        a = self.alpha(x)
        kk = self.beta(x)
        
        # Apply grounding if enabled
        if self.use_grounding:
            x, w = self.grounding(x, w)
        
        # Reshape for head-wise processing
        r = rearrange(r, 'b l (h d) -> b l h d', h=H)
        k = rearrange(k, 'b l (h d) -> b l h d', h=H)
        v = rearrange(v, 'b l (h d) -> b l h d', h=H)
        w = rearrange(w, 'b l (h d) -> b l h d', h=H)
        a = rearrange(a, 'b l (h d) -> b l h d', h=H)
        kk = rearrange(kk, 'b l (h d) -> b l h d', h=H)
        
        # Initialize state if needed
        # State is (B, H, head_dim, head_dim) - a square matrix per head
        # This maps keys to values: v ≈ k^T @ S
        if state is None:
            state = torch.zeros(B, H, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        
        # Sequential state update (for inference)
        # Training would use FLA's parallel scan kernel
        outputs = []
        for t in range(L):
            # RWKV-7 state update: S = w*S + S @ a @ kk^T + k @ v^T
            # The a @ kk^T term is the "gradient descent" learning within state
            b_kk = kk[:, t]  # (B, H, head_dim)
            b_a = a[:, t]    # (B, H, head_dim)
            b_k = k[:, t]    # (B, H, head_dim)
            b_v = v[:, t]    # (B, H, head_dim)
            b_w = w[:, t]    # (B, H, head_dim)
            b_r = r[:, t]    # (B, H, head_dim)
            
            # Compute S @ (-kk) to get the "gradient" direction
            # sa = state @ (-kk) -> (B, H, head_dim)
            sa = torch.einsum('bhij,bhj->bhi', state, -b_kk)
            
            # Update: S = w*S + sa @ (kk*a)^T + k @ v^T
            # Decay term
            state = b_w.unsqueeze(-1) * state
            # Gradient term: outer product of sa and (kk*a)
            state = state + torch.einsum('bhi,bhj->bhij', sa, b_kk * b_a)
            # Value injection: outer product of k and v
            state = state + torch.einsum('bhi,bhj->bhij', b_k, b_v)
            
            # Normalize state to prevent explosion
            state = self.state_norm(state)
            
            # Read from state
            o_t = (state * r[:, t].unsqueeze(-1)).sum(dim=-1)
            outputs.append(o_t)
        
        # Combine outputs
        output = torch.stack(outputs, dim=1)  # (B, L, H, head_dim)
        output = rearrange(output, 'b l h d -> b l (h d)')
        
        # Apply gate and output projection
        output = g * output
        output = self.out_proj(output)
        
        # Residual connection with layer norm
        output = x + self.ln(output)
        
        return output, state


class ChannelMixing(nn.Module):
    """
    RWKV-7 Channel Mixing (FFN equivalent)
    """
    
    def __init__(self, dim: int, expansion: float = 3.5):
        super().__init__()
        hidden = int(dim * expansion)
        
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        
        self.ln = RMSNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.receptance(x))
        k = torch.square(torch.relu(self.key(x)))  # Squared ReLU
        v = self.value(k)
        return x + self.ln(r * v)


class GroundThinkBlock(nn.Module):
    """Single GroundThink block = TimeMixing + ChannelMixing"""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int = 64,
        state_dim: int = 64,
        ffn_expansion: float = 3.5,
        use_grounding: bool = True,
    ):
        super().__init__()
        self.time_mixing = TimeMixing(dim, n_heads, head_dim, state_dim, use_grounding)
        self.channel_mixing = ChannelMixing(dim, ffn_expansion)
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ln1(x)
        x, state = self.time_mixing(x, state)
        x = self.ln2(x)
        x = self.channel_mixing(x)
        return x, state
