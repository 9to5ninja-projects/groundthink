import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class SelectiveRWKVBlock(nn.Module):
    """Hybrid block combining RWKV's grounding with Mamba's selection"""
    
    def __init__(self, dim=2560, n_heads=40, head_dim=64, state_dim=64, expansion=3.5):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.state_dim = state_dim
        
        # RWKV projections (grounding)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        
        # Mamba selection projections (thinking)
        self.x_proj = nn.Linear(dim, n_heads * (3 * state_dim), bias=False)  # Δ, B, C
        self.dt_proj = nn.Linear(n_heads * state_dim, dim, bias=True)
        
        # Short-term grounding (local conv)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=4,
            padding=3,
            groups=dim,
            bias=False
        )
        
        # Normalization layers
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Initialize for stability
        self._init_parameters()
    
    def _init_parameters(self):
        # Critical: Initialize selection mechanism for "medium decay"
        nn.init.constant_(self.dt_proj.bias, 1.0)  # Start with moderate memory
        
        # RWKV style initialization
        with torch.no_grad():
            # Zero-init output for stable start (O3 initialization)
            self.out_proj.weight.zero_()
            
            # Time-first gate: small positive values
            nn.init.uniform_(self.gate.weight, 0.9, 1.1)
    
    def forward(self, x, state=None):
        """
        Args:
            x: (batch, seq_len, dim)
            state: previous hidden state or None
        Returns:
            output: (batch, seq_len, dim)
            new_state: updated state for next step
        """
        B, L, D = x.shape
        
        # 1. Local grounding (short-term n-grams)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)[:, :L, :]
        x = self.ln1(x + x_conv)
        
        # 2. Project for Mamba selection
        proj = self.x_proj(x)  # (B, L, n_heads * 3 * state_dim)
        proj = rearrange(proj, 'b l (h p) -> b l h p', h=self.n_heads, p=3*self.state_dim)
        
        # Split into selection parameters
        dt_raw, B_vec, C_vec = torch.split(proj, 
                                          [self.state_dim, self.state_dim, self.state_dim], 
                                          dim=-1)
        
        # 3. Compute selective decay (Mamba thinking)
        # dt = softplus(linear(dt_raw))
        dt = F.softplus(self.dt_proj(rearrange(dt_raw, 'b l h d -> b l (h d)')))
        dt = rearrange(dt, 'b l (h d) -> b l h d', h=self.n_heads, d=self.state_dim)
        
        # Selective decay: combines RWKV stability with Mamba's input-dependence
        w = torch.exp(-dt)  # Decay factor (0 to 1)
        
        # 4. RWKV projections (grounding)
        r = torch.sigmoid(self.receptance(x))
        k = self.key(x)
        v = self.value(x)
        g = torch.sigmoid(self.gate(x))
        
        # Reshape for head-wise processing
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.n_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.n_heads)
        r = rearrange(r, 'b l (h d) -> b l h d', h=self.n_heads)
        
        # 5. Head-wise selective state update
        if state is None:
            # Initialize state if None
            state = torch.zeros(B, self.n_heads, self.state_dim, self.head_dim, 
                              device=x.device, dtype=x.dtype)
        
        outputs = []
        states = []
        
        # Sequential processing (for inference)
        # Note: Training would use parallel scan - see section 3
        for t in range(L):
            current_state = state
            
            # Update state with selective decay
            # S_t = w * S_{t-1} + B * k * v^T
            decay_factor = w[:, t].unsqueeze(-1)  # (B, h, d, 1)
            
            # B * k * v^T update (outer product)
            kv_update = einsum(B_vec[:, t], k[:, t], v[:, t], 
                             'b h s, b h d1, b h d2 -> b h s d1 d2')
            
            # Reshape kv_update to match state dimensions
            kv_update = rearrange(kv_update, 'b h s d1 d2 -> b h s (d1 d2)')
            kv_update = kv_update.view(B, self.n_heads, self.state_dim, self.head_dim)
            
            # Apply update
            new_state = decay_factor * current_state + kv_update
            states.append(new_state)
            
            # Read from state with C projection (selective recall)
            state_read = einsum(new_state, C_vec[:, t], 'b h s d, b h s -> b h d')
            
            # Apply receptance and gate
            output_t = r[:, t] * state_read * g[:, t]
            output_t = rearrange(output_t, 'b h d -> b (h d)')
            outputs.append(output_t)
            
            # Update state for next iteration
            state = new_state
        
        output = torch.stack(outputs, dim=1)
        output = self.ln2(x + self.out_proj(output))
        
        return output, state