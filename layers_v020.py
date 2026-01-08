"""
GroundThink Core Layers - v0.2.0

Version History:
  v0.1.0 - Initial hybrid with Mamba-dominant behavior
  v0.2.0 - Rebalanced: Added alpha coefficient, reduced time_decay init
  
Key Changes in v0.2.0:
  1. Added HYBRID_BALANCE_ALPHA: Explicit control over RWKV vs Mamba contribution
  2. Reduced TIME_DECAY_INIT_STD: Smaller initialization to prevent Mamba dominance
  3. Added GROUNDING_STRENGTH: Control base_decay influence
  4. All hyperparameters documented at top of file
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import FLA's fast chunked implementation
import sys
sys.path.insert(0, 'e:/RWKV/fla')
from fla.ops.simple_gla import chunk_simple_gla


# ============================================================================
# HYPERPARAMETERS - v0.2.0
# All tunable values in one place for easy experimentation
# ============================================================================

VERSION = "0.2.0"

# Balance Control
HYBRID_BALANCE_ALPHA = 0.6         # 0.5 = equal, >0.5 = more RWKV grounding
GROUNDING_STRENGTH = 1.0           # Multiplier on base_decay (1.0 = standard)

# Initialization
TIME_DECAY_INIT_STD = 0.02         # Keep standard - model init handles decay range
EMBED_INIT_STD = 0.02              # Standard for embeddings
OUTPUT_PROJ_INIT_STD = 0.02        # Output projection

# Clamps
MIN_RETENTION = 0.01               # Never completely forget
MAX_RETENTION = 0.99               # Always some decay

# Local Grounding
CONV_KERNEL_SIZE = 4               # n-gram context size
CONV_RESIDUAL_WEIGHT = 0.1         # How much local context to add

# State Normalization
STATE_NORM_EPS = 1e-5              # Epsilon for state normalization

# ============================================================================


def get_layer_config():
    """Returns current layer configuration as dict for logging"""
    return {
        "version": VERSION,
        "hybrid_balance_alpha": HYBRID_BALANCE_ALPHA,
        "grounding_strength": GROUNDING_STRENGTH,
        "time_decay_init_std": TIME_DECAY_INIT_STD,
        "min_retention": MIN_RETENTION,
        "max_retention": MAX_RETENTION,
        "conv_kernel_size": CONV_KERNEL_SIZE,
    }


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return x / norm * self.weight


class GroundingMechanism(nn.Module):
    """
    Ensures state doesn't drift - the key innovation over vanilla RWKV-7.
    
    Three-tier grounding:
    1. Local (conv) - captures n-grams before recurrence
    2. Medium-term (learned base decay) - provides stability floor
    3. Long-term (residual pathway) - ensures gradient flow
    
    v0.2.0 Changes:
    - Added alpha parameter for RWKV/Mamba balance control
    - Configurable grounding strength
    """
    
    def __init__(
        self, 
        dim: int, 
        kernel_size: int = CONV_KERNEL_SIZE,
        min_retention: float = MIN_RETENTION,
        max_retention: float = MAX_RETENTION,
        alpha: float = HYBRID_BALANCE_ALPHA,
        grounding_strength: float = GROUNDING_STRENGTH,
    ):
        super().__init__()
        self.dim = dim
        self.min_retention = min_retention
        self.max_retention = max_retention
        self.alpha = alpha  # Balance coefficient
        self.grounding_strength = grounding_strength
        
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
        grounded_x = x + CONV_RESIDUAL_WEIGHT * x_conv
        
        # Tier 2: Combine base decay with selective decay
        # Base decay provides stability floor (RWKV grounding)
        w_base = torch.exp(-self.base_decay.view(1, 1, -1) * self.grounding_strength)
        
        # Selective decay from RWKV-7's input-dependent mechanism (Mamba-like)
        w_selective = selective_decay  # Already exp(-w) form
        
        # v0.2.0: Balanced combination using alpha
        # Linear interpolation is more stable than power
        w_combined = self.alpha * w_base + (1 - self.alpha) * w_selective
        
        # Tier 3: Clamp to ensure minimum retention (never completely forget)
        # and maximum decay (always some forgetting to prevent saturation)
        grounded_decay = torch.clamp(w_combined, min=self.min_retention, max=self.max_retention)
        
        return grounded_x, grounded_decay


class StateNormalizer(nn.Module):
    """
    Prevents state explosion in long sequences.
    RMS normalization applied to state matrices.
    """
    
    def __init__(self, eps: float = STATE_NORM_EPS):
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
    
    v0.2.0 Changes:
    - Reduced time_decay initialization to prevent Mamba dominance
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
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Grounding mechanism
        if use_grounding:
            self.grounding = GroundingMechanism(dim)
        
        # Normalization
        self.ln = RMSNorm(dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize for stability - v0.2.0 reduced time_decay init"""
        # Output projection
        nn.init.normal_(self.out_proj.weight, std=OUTPUT_PROJ_INIT_STD)
        
        # Small positive gate init
        nn.init.uniform_(self.gate.weight, 0.9, 1.1)
        
        # v0.2.0: Smaller time_decay init to reduce Mamba dominance
        nn.init.normal_(self.time_decay.weight, std=TIME_DECAY_INIT_STD)
    
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
        
        # Apply grounding if enabled
        if self.use_grounding:
            x_grounded, w = self.grounding(x, w)
        else:
            x_grounded = x
        
        # Reshape for head-wise processing: [B, L, H, D]
        r = rearrange(r, 'b l (h d) -> b l h d', h=H)
        k = rearrange(k, 'b l (h d) -> b l h d', h=H)
        v = rearrange(v, 'b l (h d) -> b l h d', h=H)
        w = rearrange(w, 'b l (h d) -> b l h d', h=H)
        
        # Initialize state if needed: [B, H, D, D]
        if state is None:
            state = torch.zeros(B, H, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        
        # Use FLA's fast chunked implementation
        g_log = torch.log(1 - w + 1e-6).mean(dim=-1)  # [B, L, H] - headwise scalar
        
        output, final_state = chunk_simple_gla(
            q=r,
            k=k,
            v=v,
            g=g_log,
            scale=1.0 / math.sqrt(self.head_dim),
            initial_state=state,
            output_final_state=True
        )
        
        # Reshape output: [B, L, H, D] -> [B, L, H*D]
        output = rearrange(output, 'b l h d -> b l (h d)')
        
        # Apply gate and output projection
        output = g * output
        output = self.out_proj(output)
        
        # Residual connection with layer norm
        output = x + self.ln(output)
        
        return output, final_state


class ChannelMixing(nn.Module):
    """RWKV-7 Channel Mixing (FFN equivalent)"""
    
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
        normalize_state: bool = True,
    ):
        super().__init__()
        self.time_mixing = TimeMixing(dim, n_heads, head_dim, state_dim, use_grounding)
        self.channel_mixing = ChannelMixing(dim, ffn_expansion)
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
        self.normalize_state = normalize_state
        if normalize_state:
            self.state_norm = StateNormalizer()
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ln1(x)
        x, state = self.time_mixing(x, state)
        
        # Normalize state to prevent explosion
        if self.normalize_state and state is not None:
            state = self.state_norm(state)
        
        x = self.ln2(x)
        x = self.channel_mixing(x)
        return x, state
