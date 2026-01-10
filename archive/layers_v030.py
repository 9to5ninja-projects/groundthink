"""
⛔ DEPRECATED - SEE V3_DEPRECATED.md

This file implements RWKV-7, NOT the requested RWKV-6 + Mamba-2.
Do not use for V4 development.

---

GroundThink V3 Core Layers - v0.3.0 (DEPRECATED)

Implements the V3 architecture per V3_RESEARCH_NOTES.md specifications:
- StateNorm (Grouped) - Section 2.6, 2.11
- HybridBlock (Parallel Residual) - Section 2.21
- HybridStack (Attention Anchors) - Section 2.20
- Identity-SSM Initialization - Section 2.12
- Trainable h0 - Section 2.13

Version History:
  v0.2.0 - Working hybrid with FLA integration
  v0.3.0 - V3 architecture: StateNorm, Parallel Residual, Identity Init
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

VERSION = "0.3.0"

# ============================================================================
# HYPERPARAMETERS - v0.3.0
# ============================================================================

# Hybrid Balance (from v0.2.0, validated)
HYBRID_BALANCE_ALPHA = 0.6
GROUNDING_STRENGTH = 1.0
MIN_RETENTION = 0.01
MAX_RETENTION = 0.99

# V3 Additions
STATE_NORM_GROUPS = 4          # Independent sub-states for stability
RESIDUAL_GAMMA_INIT = 0.01     # Identity-preserving residual scaling
CONV_KERNEL_SIZE = 4
CONV_RESIDUAL_WEIGHT = 0.1

# ============================================================================


def get_layer_config():
    """Returns current layer configuration for logging"""
    return {
        "version": VERSION,
        "hybrid_balance_alpha": HYBRID_BALANCE_ALPHA,
        "state_norm_groups": STATE_NORM_GROUPS,
        "residual_gamma_init": RESIDUAL_GAMMA_INIT,
        "min_retention": MIN_RETENTION,
        "max_retention": MAX_RETENTION,
    }


# ============================================================================
# NORMALIZATION LAYERS
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class StateNorm(nn.Module):
    """
    Grouped State Normalization - V3 Core Component
    
    From V3_RESEARCH_NOTES.md Section 2.6/2.11:
    - RMSNorm applied to hidden state inside recurrent loop
    - Groups allows for 'Grouped State Norm' - different heads track different things
    - Prevents state saturation that causes the 7.0 loss wall
    
    Args:
        n_embd: Hidden dimension
        groups: Number of independent sub-states (default 4)
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, n_embd: int, groups: int = STATE_NORM_GROUPS, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.groups = groups
        self.n_embd = n_embd
        # Learnable gain for each channel
        self.weight = nn.Parameter(torch.ones(n_embd))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: State tensor, shape [B, ...] with last dim = n_embd
        Returns:
            Normalized state, same shape
        """
        orig_shape = x.shape
        
        if self.groups > 1 and x.shape[-1] == self.n_embd:
            # Reshape to [B, ..., groups, n_embd // groups]
            x = x.view(*x.shape[:-1], self.groups, -1)
            # RMS norm along last dim (within each group)
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x = x * norm
            # Reshape back
            x = x.view(orig_shape)
        else:
            # Standard RMS norm across all dimensions
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x = x * norm
        
        return x * self.weight


# ============================================================================
# GROUNDING MECHANISM (from v0.2.0)
# ============================================================================

class GroundingMechanism(nn.Module):
    """
    Ensures state doesn't drift - key innovation over vanilla RWKV-7.
    (Carried forward from v0.2.0 with minor updates)
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
        self.alpha = alpha
        self.grounding_strength = grounding_strength
        
        # Local grounding via depthwise conv
        self.conv_short = nn.Conv1d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=kernel_size - 1,
            groups=dim,
            bias=False
        )
        
        # Learned base decay (stability floor)
        self.base_decay = nn.Parameter(torch.ones(dim))
        
        # Residual weight for long-term grounding
        self.res_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, selective_decay: torch.Tensor) -> tuple:
        B, L, D = x.shape
        
        # Local n-gram grounding
        x_conv = self.conv_short(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        grounded_x = x + CONV_RESIDUAL_WEIGHT * x_conv
        
        # Combine base decay with selective decay
        w_base = torch.exp(-self.base_decay.view(1, 1, -1) * self.grounding_strength)
        w_selective = selective_decay
        
        # Balanced combination using alpha
        w_combined = self.alpha * w_base + (1 - self.alpha) * w_selective
        
        # Clamp for stability
        grounded_decay = torch.clamp(w_combined, min=self.min_retention, max=self.max_retention)
        
        return grounded_x, grounded_decay


# ============================================================================
# TIME MIXING (RWKV-7 Style with Grounding)
# ============================================================================

class TimeMixing(nn.Module):
    """
    RWKV-7 Time Mixing with Grounding and StateNorm.
    Uses FLA's chunk_simple_gla for GPU efficiency.
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
        
        # Time decay (input-dependent)
        self.time_decay = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Grounding mechanism
        if use_grounding:
            self.grounding = GroundingMechanism(dim)
        
        # V3: StateNorm inside recurrent loop
        self.state_norm = StateNorm(head_dim * head_dim, groups=STATE_NORM_GROUPS)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.uniform_(self.gate.weight, 0.9, 1.1)
        nn.init.normal_(self.time_decay.weight, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        H = self.n_heads
        
        # Project inputs
        r = torch.sigmoid(self.receptance(x))
        k = self.key(x)
        v = self.value(x)
        g = torch.sigmoid(self.gate(x))
        
        # Input-dependent decay
        w = self.time_decay(x)
        w = torch.exp(-torch.exp(w))
        
        # Apply grounding if enabled
        if self.use_grounding:
            x_grounded, w = self.grounding(x, w)
        
        # Reshape for head-wise processing
        r = rearrange(r, 'b l (h d) -> b l h d', h=H)
        k = rearrange(k, 'b l (h d) -> b l h d', h=H)
        v = rearrange(v, 'b l (h d) -> b l h d', h=H)
        w = rearrange(w, 'b l (h d) -> b l h d', h=H)
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(B, H, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        
        # Use FLA's fast chunked implementation
        g_log = torch.log(1 - w + 1e-6).mean(dim=-1)
        
        # Note: FLA warning about T < H is a false positive for single-token inference
        # Verified 2026-01-08: shapes are correct [B, T, H, D], kernel updates state properly
        
        output, final_state = chunk_simple_gla(
            q=r, k=k, v=v, g=g_log,
            scale=1.0 / math.sqrt(self.head_dim),
            initial_state=state,
            output_final_state=True
        )
        
        # V3: Apply StateNorm to final state
        B_s, H_s, D1, D2 = final_state.shape
        final_state_flat = final_state.view(B_s, H_s, -1)
        final_state_flat = self.state_norm(final_state_flat)
        final_state = final_state_flat.view(B_s, H_s, D1, D2)
        
        # Reshape output
        output = rearrange(output, 'b l h d -> b l (h d)')
        
        # Apply gate and output projection
        output = g * output
        output = self.out_proj(output)
        
        return output, final_state


# ============================================================================
# CHANNEL MIXING (FFN)
# ============================================================================

class ChannelMixing(nn.Module):
    """RWKV-7 Channel Mixing (FFN equivalent)"""
    
    def __init__(self, dim: int, expansion: float = 3.5):
        super().__init__()
        hidden = int(dim * expansion)
        
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.receptance(x))
        k = torch.square(torch.relu(self.key(x)))
        v = self.value(k)
        return r * v


# ============================================================================
# HYBRID BLOCK - V3 Core (Parallel Residual)
# ============================================================================

class HybridBlock(nn.Module):
    """
    V3 Hybrid Block with Parallel Residual Architecture
    
    From V3_RESEARCH_NOTES.md Section 2.21:
    - Parallel path: mamba_out + attn_out (not sequential)
    - Gamma residual scaling (0.01) for identity-preserving start
    - Pre-norm architecture
    
    The parallel design maintains higher gradient SNR and prevents
    the "7.0 loss wall" by allowing identity to pass through during
    early training.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int = 32,
        state_dim: int = 16,
        ffn_expansion: float = 3.5,
        use_grounding: bool = True,
        gamma_init: float = RESIDUAL_GAMMA_INIT,
    ):
        super().__init__()
        self.dim = dim
        
        # Pre-norm
        self.ln_1 = RMSNorm(dim)
        self.ln_2 = RMSNorm(dim)
        
        # Time mixing (RWKV/Mamba path)
        self.time_mixing = TimeMixing(dim, n_heads, head_dim, state_dim, use_grounding)
        
        # Channel mixing (FFN)
        self.channel_mixing = ChannelMixing(dim, ffn_expansion)
        
        # V3: Learnable residual scaling (gamma)
        # Initialized to 0.01 for identity-preserving start
        self.gamma_1 = nn.Parameter(torch.ones(dim) * gamma_init)
        self.gamma_2 = nn.Parameter(torch.ones(dim) * gamma_init)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with parallel residual.
        
        Args:
            x: Input tensor [B, L, D]
            state: Previous hidden state or None
            
        Returns:
            output: [B, L, D]
            new_state: Updated hidden state
        """
        # Time mixing with residual scaling
        norm_x = self.ln_1(x)
        tm_out, new_state = self.time_mixing(norm_x, state)
        x = x + self.gamma_1 * tm_out
        
        # Channel mixing with residual scaling
        norm_x = self.ln_2(x)
        cm_out = self.channel_mixing(norm_x)
        x = x + self.gamma_2 * cm_out
        
        return x, new_state


# ============================================================================
# ATTENTION LAYER (for Hybrid Stack anchors)
# ============================================================================

class SimpleAttention(nn.Module):
    """
    Simple multi-head attention for use as "attention anchor" in hybrid stack.
    From Section 2.20: 1:5 ratio of attention to recurrent layers.
    """
    
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.n_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.n_heads)
        
        # Causal attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.out_proj(out)


class AttentionBlock(nn.Module):
    """Attention block for anchor positions in hybrid stack."""
    
    def __init__(self, dim: int, n_heads: int, ffn_expansion: float = 3.5):
        super().__init__()
        self.ln_1 = RMSNorm(dim)
        self.ln_2 = RMSNorm(dim)
        self.attn = SimpleAttention(dim, n_heads)
        self.ffn = ChannelMixing(dim, ffn_expansion)
        self.gamma_1 = nn.Parameter(torch.ones(dim) * RESIDUAL_GAMMA_INIT)
        self.gamma_2 = nn.Parameter(torch.ones(dim) * RESIDUAL_GAMMA_INIT)
    
    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        # Attention doesn't use state - return None to avoid frozen noise anchor
        # Per V3_RESEARCH_NOTES: Attention refreshes hidden state via global context,
        # but does not store recurrent state. Returning None signals to autograd
        # that there is no state dependency to track for this layer.
        x = x + self.gamma_1 * self.attn(self.ln_1(x))
        x = x + self.gamma_2 * self.ffn(self.ln_2(x))
        return x, None


# ============================================================================
# HYBRID STACK - V3 Full Model Stack
# ============================================================================

class HybridStack(nn.Module):
    """
    V3 Hybrid Stack with Attention Anchors
    
    From V3_RESEARCH_NOTES.md Section 2.20:
    - 1:5 or 1:7 ratio of attention to recurrent layers
    - Attention at middle and/or final layer for "global reset"
    - Mamba/RWKV layers build representation, attention refreshes state
    """
    
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        head_dim: int = 32,
        state_dim: int = 16,
        attn_positions: list[int] | None = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        
        # Default: attention at middle layer only (1:N ratio)
        if attn_positions is None:
            attn_positions = [n_layers // 2]
        self.attn_positions = set(attn_positions)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in self.attn_positions:
                self.layers.append(AttentionBlock(dim, n_heads))
            else:
                self.layers.append(HybridBlock(dim, n_heads, head_dim, state_dim))
        
        n_attn = len(self.attn_positions)
        n_recurrent = n_layers - n_attn
        print(f"HybridStack: {n_attn} Attention, {n_recurrent} Recurrent (1:{n_recurrent//max(1,n_attn)} ratio)")
    
    def forward(
        self, 
        x: torch.Tensor, 
        states: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward through all layers.
        
        Args:
            x: Input [B, L, D]
            states: List of states for each layer, or None
            
        Returns:
            output: [B, L, D]
            new_states: List of updated states
        """
        if states is None:
            states = [None] * self.n_layers
        
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, states[i])
            new_states.append(state)
        
        return x, new_states


# ============================================================================
# INITIALIZATION FUNCTIONS - V3 Identity-SSM Init
# ============================================================================

def senior_init_hybrid_state(model: nn.Module) -> None:
    """
    Phase 2.1: Senior Identity Initialization per V3_RESEARCH_NOTES.md Section 2.12
    
    This initialization breaks the 7.0 loss wall by:
    1. Initializing A_log in structured log-space for spectral radius control
    2. Setting decay/gate vectors to 0 (multiplicative 1) for "perfect memory" start
    3. Using small gain (0.1) for B_proj and C_proj to keep signal variance 1.0
    """
    init_count = 0
    for name, param in model.named_parameters():
        # 1. THE A-MATRIX (Mamba/SSM Transition)
        # We want exp(delta * A) ≈ I. Use structured log-space init.
        if "A_log" in name:
            with torch.no_grad():
                # Range from 1 to N (state size) - different 'memory speeds' for channels
                a_values = torch.arange(1, param.shape[-1] + 1, dtype=param.dtype, device=param.device)
                param.copy_(torch.log(a_values).expand_as(param))
                init_count += 1

        # 2. THE HOUSEHOLDER TRICK (RWKV-7 Secret)
        # Initialize gating to 1.0 (0.0 in log-space)
        elif "state_gate" in name or "decay_vector" in name or "time_decay" in name or "base_decay" in name:
            with torch.no_grad():
                # 0.0 in log space = 1.0 multiplier = no decay initially
                nn.init.constant_(param, 0.0)
                init_count += 1

        # 3. THE 'IDENTITY' PROJECTION (B & C Matrices)
        # Use Xavier with small gain to keep signal variance 1.0
        elif ("B_proj" in name or "C_proj" in name) and "weight" in name:
            with torch.no_grad():
                nn.init.xavier_uniform_(param, gain=0.1)
                init_count += 1

        # 4. Output projections also use small gain
        elif 'out_proj' in name and 'weight' in name:
            with torch.no_grad():
                nn.init.xavier_uniform_(param, gain=0.1)
                init_count += 1

    print(f"[senior_init_hybrid_state] Initialized {init_count} parameters")


def init_identity_bias(model: nn.Module) -> None:
    """
    Phase 2.2: Identity-Bias Initialization per V3_RESEARCH_NOTES.md Section 2.26
    
    Initialize decay parameters for long-memory bias (Recency-Bias).
    Forces model to preserve early tokens (system prompt) by default.
    
    Decay = 0.9999 means signal retains 95% after 512 tokens (0.9999^512 ≈ 0.95)
    """
    import math
    init_count = 0
    for name, param in model.named_parameters():
        if 'time_decay' in name or 'A_log' in name or 'base_decay' in name:
            with torch.no_grad():
                # 0.9999 decay = extremely long memory
                param.fill_(math.log(0.9999))
                init_count += 1

    print(f"[init_identity_bias] Applied long-memory bias to {init_count} parameters")


def identity_ssm_init(model: nn.Module, use_identity_bias: bool = False) -> None:
    """
    Combined initialization function for convenience.
    
    Args:
        model: The model to initialize
        use_identity_bias: If True, use Section 2.26 init (long memory with 0.9999 decay)
                          If False, use Section 2.12 init (perfect memory with 0.0 decay)
    """
    senior_init_hybrid_state(model)
    if use_identity_bias:
        init_identity_bias(model)  # Overwrites decay params with 0.9999 bias


def get_optimizer_groups(model: nn.Module, lr_base: float = 6e-4, weight_decay: float = 0.1):
    """
    Create optimizer parameter groups per V3_RESEARCH_NOTES.md Section 2.15/2.31
    
    Groups:
    1. Standard weights (projections, FFNs) - normal LR, with decay
    2. Recurrent states (A, gates, decay) - lower LR, no decay
    3. Norms, biases, gamma - normal LR, no decay
    """
    # Exclusion patterns (no weight decay)
    no_decay_patterns = [
        'bias', 'norm', 'ln', 'A_log', 'time_decay', 'base_decay',
        'delta', 'gate', 'gamma', 'h0', 'embed', 'res_weight'
    ]
    
    # Recurrent patterns (lower LR)
    recurrent_patterns = ['time_decay', 'base_decay', 'gate', 'grounding']
    
    decay_params = []
    no_decay_params = []
    recurrent_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        is_no_decay = any(p in name.lower() for p in no_decay_patterns)
        is_recurrent = any(p in name.lower() for p in recurrent_patterns)
        
        if is_recurrent:
            recurrent_params.append(param)
        elif is_no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr_base, 'name': 'decay'},
        {'params': recurrent_params, 'weight_decay': 0.0, 'lr': lr_base * 0.1, 'name': 'recurrent'},
        {'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr_base, 'name': 'no_decay'},
    ]


# ============================================================================
# FULL MODEL - V3 GroundThink
# ============================================================================

class GroundThinkV3(nn.Module):
    """
    GroundThink V3 Model
    
    Features:
    - Trainable h0 (learnable initial state) per Section 2.13
    - HybridStack with attention anchors per Section 2.20
    - Identity-SSM initialization per Section 2.12
    - Tied embeddings for semantic symmetry
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        n_layers: int = 12,
        dim: int = 256,
        n_heads: int = 8,
        head_dim: int = 32,
        state_dim: int = 16,
        attn_positions: list[int] | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.state_dim = state_dim
        
        # Embedding
        self.embed = nn.Embedding(vocab_size, dim)
        
        # V3: Trainable initial state (h0)
        # "The model's baseline personality"
        self.h0 = nn.Parameter(torch.randn(n_layers, n_heads, head_dim, head_dim) * 0.01)
        
        # Hybrid stack
        self.stack = HybridStack(n_layers, dim, n_heads, head_dim, state_dim, attn_positions)
        
        # Output
        self.ln_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.head.weight = self.embed.weight
        
        # Apply initialization
        self._init_weights()
        identity_ssm_init(self)
    
    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
    
    def get_initial_states(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        """Get initial states from trainable h0."""
        states = []
        for i in range(self.n_layers):
            # Expand h0 to batch size
            state = self.h0[i].unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
            states.append(state)
        return states
    
    def forward(
        self, 
        x: torch.Tensor, 
        states: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input token ids [B, L]
            states: Optional list of states per layer
            
        Returns:
            logits: [B, L, vocab_size]
            new_states: List of updated states
        """
        B, L = x.shape
        
        # Get initial states if not provided
        if states is None:
            states = self.get_initial_states(B, x.device)
        
        # Embed
        x = self.embed(x)
        
        # Forward through stack
        x, new_states = self.stack(x, states)
        
        # Output
        x = self.ln_out(x)
        logits = self.head(x)
        
        return logits, new_states
    
    def get_param_groups(self, lr_base: float = 6e-4, weight_decay: float = 0.1):
        """Get optimizer parameter groups."""
        return get_optimizer_groups(self, lr_base, weight_decay)
