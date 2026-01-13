"""
Mamba-2 SSD Minimal Prototype (Pure PyTorch)

Adapted from state-spaces/mamba ssd_minimal.py (Apache-2.0)
No CUDA dependencies - suitable for Colab free tier.

For Task 0.0.2: Mamba-2 variance characterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def segsum(x):
    """
    Stable segment sum for cumulative decay.
    Input x: (..., T) 
    Output: (..., T, T) lower triangular cumsum
    """
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)  # (..., T, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal(X, A, B, C, block_len, initial_states=None):
    """
    Minimal SSD (State Space Duality) - Mamba-2 core algorithm.
    
    Args:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads) - decay
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        block_len: chunk size for efficient computation
        initial_states: optional (batch, n_heads, d_head, d_state)
    
    Returns:
        Y: (batch, length, n_heads, d_head)
        final_state: (batch, n_heads, d_head, d_state)
    """
    batch, seqlen, nheads, headdim = X.shape
    dstate = B.shape[-1]
    
    # Reshape into chunks
    X = rearrange(X, "b (c l) h p -> b c l h p", l=block_len)
    A = rearrange(A, "b (c l) h -> b c l h", l=block_len)
    B = rearrange(B, "b (c l) h n -> b c l h n", l=block_len)
    C = rearrange(C, "b (c l) h n -> b c l h n", l=block_len)
    
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    
    # 1. Intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    
    # 2. State at chunk boundaries
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    
    # 3. Inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    # A_cumsum is (B, H, C, L), take last position of each chunk
    A_last = A_cumsum[:, :, :, -1]  # (B, H, C)
    decay_chunk = torch.exp(segsum(F.pad(A_last, (1, 0))))  # (B, H, C+1, C+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    
    # 4. State to output
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)
    
    # Combine
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


class Mamba2TimeMix(nn.Module):
    """
    Mamba-2 time-mixing layer for variance characterization.
    Analogous to RWKV6TimeMix - just the SSM, no FFN.
    """
    
    def __init__(self, d_model, d_state=64, n_heads=4, chunk_size=64):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_state = d_state
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        
        # Projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)  # X, gate
        self.dt_proj = nn.Linear(d_model, n_heads, bias=True)
        self.B_proj = nn.Linear(d_model, n_heads * d_state, bias=False)
        self.C_proj = nn.Linear(d_model, n_heads * d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable decay (A is negative)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 16, n_heads)))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.dt_proj.bias)
    
    def forward(self, x, state=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            state: optional previous state
        Returns:
            y: (batch, seq_len, d_model)
            new_state: updated state
        """
        batch, seqlen, _ = x.shape
        
        # Pad to chunk boundary
        pad_len = (self.chunk_size - seqlen % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        # Projections
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        dt = F.softplus(self.dt_proj(x))  # (B, L, H)
        A = -torch.exp(self.A_log)  # (H,)
        A = A[None, None, :] * dt  # (B, L, H)
        
        B = self.B_proj(x).view(batch, -1, self.n_heads, self.d_state)
        C = self.C_proj(x).view(batch, -1, self.n_heads, self.d_state)
        X = x_in.view(batch, -1, self.n_heads, self.d_head)
        
        # Apply dt scaling
        X = X * dt.unsqueeze(-1)
        
        # SSD core
        Y, new_state = ssd_minimal(X, A, B, C, self.chunk_size, state)
        
        # Gate and project
        y = Y.reshape(batch, -1, self.d_model)
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        # Remove padding
        if pad_len > 0:
            y = y[:, :seqlen]
        
        return y, new_state


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)
    layer = Mamba2TimeMix(d_model=128, n_heads=4, d_state=32, chunk_size=32)
    x = torch.randn(2, 64, 128)
    y, state = layer(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Input var: {x.var():.4f}, Output var: {y.var():.4f}")
    print(f"Ratio: {(y.var() / x.var()):.4f}")
