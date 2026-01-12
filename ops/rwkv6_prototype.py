"""
RWKV-6 PyTorch Prototype - Time-Mixing Block

⚠️ WARNING: This is a VALIDATION-ONLY prototype, NOT production RWKV-6!

Purpose: Minimum viable RWKV-6 implementation for prototype validation.
This serves as the specification reference for building the CUDA kernel wrapper.

⚠️ CRITICAL ARCHITECTURE NOTES:
1. RWKV6Attention_Prototype is a COMPLETE BLOCK (not just attention)
   - Includes: LayerNorm → Time-Mixing → Residual → LayerNorm → FFN → Residual
   - Do NOT wrap with additional LayerNorm or FFN!
   - Stack directly: [RWKV6Attention(...) for _ in range(num_layers)]

2. WKV Computation is NORMALIZED:
   - wkv = numerator / denominator (proper weighted average)
   - Previous bug: accumulated exp(k)*v without normalization → unbounded output
   - Fixed 2026-01-11: Tracks state_num and state_den separately

Key Features:
- ✅ Proper _wkv_sequential() - Implements recurrent state update with normalization
- ✅ Time decay parameters used in computation (not just placeholders)  
- ✅ Squared ReLU for channel mixing (RWKV spec detail)
- ✅ Returns tuple format: (output, None, None)
- ⚠️ Sequential loop = O(B*T), not optimized (for validation only)
- ⚠️ ~50x slower than CUDA kernel (acceptable for <1K steps)

Performance Notes:
- Suitable for validation with seq_len < 512, steps < 1000
- For production: use CUDA kernel from RWKV-CUDA/wkv6/
- Colab free tier: ~0.5s/step (CPU), acceptable for baseline characterization

Reference: V4.5_PYTHON_WRAPPERS.md, V4_HANDOFF.md (deviations section)
Created: 2026-01-09
Updated: 2026-01-11 (WKV normalization fix, architecture clarification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RWKV6Attention_Prototype(nn.Module):
    """
    PyTorch-only RWKV-6 time-mixing block for prototype validation.
    
    This implementation captures the mathematical essence of RWKV-6:
    - WKV recurrence: state_t = decay * state_{t-1} + k_t * v_t
    - Receptance gating: output = sigmoid(r) * wkv
    - Time decay with learned parameters
    
    Args:
        hidden_size: Model dimension (d_model)
        n_head: Number of attention heads (default: 4)
        head_size: Dimension per head (default: 64)
    """
    
    def __init__(self, hidden_size, n_head=None, head_size=64, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Compute n_head from hidden_size and head_size if not provided
        if n_head is None:
            n_head = hidden_size // head_size
        
        self.n_head = n_head
        self.head_size = head_size
        
        # Validate dimensions
        assert hidden_size == n_head * head_size, \
            f"hidden_size ({hidden_size}) must equal n_head ({n_head}) * head_size ({head_size})"
        
        # Time-mixing parameters (core RWKV-6 feature)
        # time_decay: learned decay rate per head/position
        # time_first: bonus for current token (u in RWKV notation)
        self.time_decay = nn.Parameter(torch.ones(n_head, head_size) * -5.0)  # Start with moderate decay
        self.time_first = nn.Parameter(torch.zeros(n_head, head_size))
        
        # Linear projections (no bias as per RWKV spec)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Channel-mixing (FFN-like, after time mixing)
        self.ffn_key = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.ffn_value = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        
        # Layer normalization (RWKV uses PreLN)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Initialize projections
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following RWKV conventions"""
        for module in [self.key, self.value, self.receptance, self.output]:
            nn.init.orthogonal_(module.weight, gain=1.0)
        nn.init.orthogonal_(self.ffn_key.weight, gain=1.0)
        nn.init.orthogonal_(self.ffn_value.weight, gain=1.0)
        
    def forward(self, x, state=None, return_state=False):
        """
        Forward pass through RWKV-6 time-mixing block.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            state: Optional recurrent state (not used in this prototype)
            return_state: If True, return internal recurrent state for state extraction
            
        Returns:
            If return_state=False: Tuple of (output, None, None) - matches expected interface
            If return_state=True: Tuple of (output, None, state_dict) where state_dict contains
                                  'rwkv_state' with shape [B, H, S]
        """
        B, T, C = x.shape
        H, S = self.n_head, self.head_size
        
        # ========== Time-Mixing Block ==========
        x_ln = self.ln1(x)
        
        # Project to key, value, receptance
        # Reshape for multi-head: [B, T, C] -> [B, T, H, S] -> [B, H, T, S]
        k = self.key(x_ln).view(B, T, H, S).transpose(1, 2)  # [B, H, T, S]
        v = self.value(x_ln).view(B, T, H, S).transpose(1, 2)
        r = self.receptance(x_ln).view(B, T, H, S).transpose(1, 2)
        
        # RWKV recurrence computation (pure PyTorch, non-optimized)
        # This is the core WKV kernel in sequential form
        wkv, final_state = self._wkv_sequential(k, v)  # [B, H, T, S], [B, H, S]
        
        # Apply receptance gating and output projection
        # receptance acts as a learned gate (sigmoid applied)
        rwkv = torch.sigmoid(r) * wkv
        rwkv = rwkv.transpose(1, 2).contiguous().view(B, T, C)
        output1 = self.output(rwkv)
        
        # Residual connection
        x = x + output1
        
        # ========== Channel-Mixing Block ==========
        x_ln = self.ln2(x)
        # Squared ReLU is an RWKV design choice for channel mixing
        ffn_out = self.ffn_value(F.relu(self.ffn_key(x_ln)) ** 2)
        x = x + ffn_out
        
        if return_state:
            # Return state dict for S0-S4 tests
            state_dict = {'rwkv_state': final_state}  # [B, H, S]
            return x, None, state_dict
        
        return x, None, None  # Match expected tuple format
    
    def _wkv_sequential(self, k, v):
        """
        Sequential WKV computation - correct but slow (for validation only).
        
        This implements the RWKV-6 recurrence with proper normalization:
            numerator_t = decay * numerator_{t-1} + exp(k_t) * v_t
            denominator_t = decay * denominator_{t-1} + exp(k_t)
            output_t = numerator_t / denominator_t
        
        In the actual CUDA kernel (wkv6_cuda_v1.cu), this is parallelized
        using associative scan techniques.
        
        Args:
            k: Keys [B, H, T, S]
            v: Values [B, H, T, S]
            
        Returns:
            wkv: Weighted key-value output [B, H, T, S]
            final_state: Final recurrent state [B, H, S] for state extraction
        """
        B, H, T, S = k.shape
        device = k.device
        dtype = k.dtype
        
        # Prepare time decay and first
        # exp(-exp(w)) gives decay in (0, 1), faster for more negative w
        time_decay = torch.exp(-torch.exp(self.time_decay.float()))  # [H, S]
        time_first = torch.exp(self.time_first.float())  # [H, S], bonus for first token
        
        # Initialize output
        wkv = torch.zeros(B, H, T, S, device=device, dtype=torch.float32)
        
        # Initialize state (numerator and denominator for proper normalization)
        state_num = torch.zeros(B, H, S, device=device, dtype=torch.float32)
        state_den = torch.zeros(B, H, S, device=device, dtype=torch.float32)
        
        for t in range(T):
            # Current token's contribution
            kt = k[:, :, t].float()  # [B, H, S]
            vt = v[:, :, t].float()  # [B, H, S]
            
            # Weight for current token (exp(k) to make it positive)
            wt = torch.exp(kt.clamp(max=30))  # Clamp to prevent overflow [B, H, S]
            
            if t == 0:
                # First token: use time_first bonus
                state_num = time_first * wt * vt
                state_den = time_first * wt
            else:
                # Recurrence: decay previous state, add new contribution
                state_num = time_decay * state_num + wt * vt
                state_den = time_decay * state_den + wt
            
            # Normalized output (prevents unbounded growth)
            wkv[:, :, t] = state_num / (state_den + 1e-6)
        
        # Return both output and final state (use numerator for state tests)
        return wkv.to(dtype), state_num.to(dtype)


# Alias for compatibility with hybrid_v4.py import expectations
RWKV6Attention = RWKV6Attention_Prototype


class RWKV6TimeMix(nn.Module):
    """
    RWKV-6 Time-Mixing ONLY (no FFN, no internal LN/residuals).
    
    Use this when you want to wrap with your own FFN and normalization.
    This is just the WKV attention mechanism.
    
    Usage:
        class MyBlock(nn.Module):
            def __init__(self, hidden):
                self.ln1 = nn.LayerNorm(hidden)
                self.time_mix = RWKV6TimeMix(hidden, num_heads=4)
                self.ln2 = nn.LayerNorm(hidden)
                self.ffn = nn.Sequential(...)
            
            def forward(self, x):
                x = x + self.time_mix(self.ln1(x))
                x = x + self.ffn(self.ln2(x))
                return x
    """
    
    def __init__(self, hidden_size, num_heads=4, layer_idx=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, f"hidden_size must be divisible by num_heads"
        
        # Time-mixing parameters
        self.time_decay = nn.Parameter(torch.ones(num_heads, self.head_size) * -5.0)
        self.time_first = nn.Parameter(torch.zeros(num_heads, self.head_size))
        
        # Linear projections
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Conservative initialization
        for m in [self.key, self.value, self.receptance, self.output]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C] (should be layer-normalized)
            
        Returns:
            output: Time-mixed output [B, T, C]
        """
        B, T, C = x.shape
        H, S = self.num_heads, self.head_size
        
        # Project to key, value, receptance
        k = self.key(x).view(B, T, H, S).transpose(1, 2)
        v = self.value(x).view(B, T, H, S).transpose(1, 2)
        r = self.receptance(x).view(B, T, H, S).transpose(1, 2)
        
        # WKV computation with normalization
        wkv = self._wkv_sequential(k, v)
        
        # Apply receptance gating
        rwkv = torch.sigmoid(r) * wkv
        rwkv = rwkv.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.output(rwkv)
    
    def _wkv_sequential(self, k, v):
        """Sequential WKV with proper normalization"""
        B, H, T, S = k.shape
        device, dtype = k.device, k.dtype
        
        time_decay = torch.exp(-torch.exp(self.time_decay.float()))
        time_first = torch.exp(self.time_first.float())
        
        wkv = torch.zeros(B, H, T, S, device=device, dtype=torch.float32)
        state_num = torch.zeros(B, H, S, device=device, dtype=torch.float32)
        state_den = torch.zeros(B, H, S, device=device, dtype=torch.float32)
        
        for t in range(T):
            kt = k[:, :, t].float()
            vt = v[:, :, t].float()
            wt = torch.exp(kt.clamp(-10, 10))  # Tighter clamp for stability
            
            if t == 0:
                state_num = time_first * wt * vt
                state_den = time_first * wt
            else:
                state_num = time_decay * state_num + wt * vt
                state_den = time_decay * state_den + wt
            
            wkv[:, :, t] = state_num / (state_den + 1e-6)
        
        return wkv.to(dtype)


def test_rwkv6_prototype():
    """Quick test for G1 gate: forward pass with no NaN"""
    print("Testing RWKV6Attention_Prototype...")
    
    # Create model
    hidden_size = 256
    model = RWKV6Attention_Prototype(hidden_size, n_head=4, head_size=64)
    
    # Test on CPU first
    x = torch.randn(2, 32, hidden_size)
    output, _, _ = model(x)
    
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    assert not torch.isnan(output).any(), "NaN detected in output"
    
    print(f"  ✓ CPU forward pass: {x.shape} -> {output.shape}")
    print(f"  ✓ No NaN: True")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        output, _, _ = model(x)
        
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        assert not torch.isnan(output).any(), "NaN detected in output"
        
        print(f"  ✓ GPU forward pass: {x.shape} -> {output.shape}")
        print(f"  ✓ No NaN: True")
    
    print("RWKV6Attention_Prototype: All tests passed!")
    return True


if __name__ == "__main__":
    test_rwkv6_prototype()
