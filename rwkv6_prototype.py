"""
RWKV-6 PyTorch Prototype - Time-Mixing Block

Purpose: Minimum viable RWKV-6 implementation for prototype validation.
This serves as the specification reference for building the CUDA kernel wrapper.

Key Features:
- ✅ Proper _wkv_sequential() - Implements recurrent state update
- ✅ Time decay parameters used in computation (not just placeholders)  
- ✅ Squared ReLU for channel mixing (RWKV spec detail)
- ✅ Returns tuple format: (output, None, None)
- ⚠️ Sequential loop = O(B*T), not optimized (for validation only)

Performance Notes:
- Suitable for validation with seq_len < 512
- For production: use CUDA kernel from RWKV-CUDA/wkv6/

Reference: V4.5_PYTHON_WRAPPERS.md
Created: 2026-01-09
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
        
    def forward(self, x, state=None):
        """
        Forward pass through RWKV-6 time-mixing block.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            state: Optional recurrent state (not used in this prototype)
            
        Returns:
            Tuple of (output, None, None) - matches expected interface
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
        wkv = self._wkv_sequential(k, v)  # [B, H, T, S]
        
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
        
        return x, None, None  # Match expected tuple format
    
    def _wkv_sequential(self, k, v):
        """
        Sequential WKV computation - correct but slow (for validation only).
        
        This implements the RWKV-6 recurrence:
            state_t = exp(-exp(time_decay)) * state_{t-1} + exp(k_t) * v_t
            output_t = sum_j(state_t[j]) (weighted by time_first for t=0)
        
        In the actual CUDA kernel (wkv6_cuda_v1.cu), this is parallelized
        using associative scan techniques.
        
        Args:
            k: Keys [B, H, T, S]
            v: Values [B, H, T, S]
            
        Returns:
            wkv: Weighted key-value output [B, H, T, S]
        """
        B, H, T, S = k.shape
        device = k.device
        dtype = k.dtype
        
        # Prepare time decay and first
        # exp(-exp(w)) gives decay in (0, 1), faster for more negative w
        time_decay = torch.exp(-torch.exp(self.time_decay.float()))  # [H, S]
        time_first = self.time_first.float()  # [H, S], bonus for current token
        
        # Initialize output
        wkv = torch.zeros(B, H, T, S, device=device, dtype=torch.float32)
        
        # Initialize state (accumulator for weighted k*v)
        state = torch.zeros(B, H, S, device=device, dtype=torch.float32)
        
        for t in range(T):
            # Current token's contribution
            kt = k[:, :, t].float()  # [B, H, S]
            vt = v[:, :, t].float()  # [B, H, S]
            
            # Weight for current token (exp(k) to make it positive)
            # In RWKV, larger k means more weight to current token
            wt = torch.exp(kt)  # [B, H, S]
            
            if t == 0:
                # First token: use time_first bonus
                wkv[:, :, t] = wt * vt + torch.exp(time_first) * vt
                state = wt * vt
            else:
                # Recurrence: decay previous state, add new contribution
                # state_t = decay * state_{t-1} + w_t * v_t
                state = time_decay * state + wt * vt
                wkv[:, :, t] = state
        
        return wkv.to(dtype)


# Alias for compatibility with hybrid_v4.py import expectations
RWKV6Attention = RWKV6Attention_Prototype


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
