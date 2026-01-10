"""
Replacements for FLA modules that your hybrid_v4.py needs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== RWKV6 Replacement ==========
class RWKV6Attention(nn.Module):
    """Replacement for fla.layers.rwkv6.RWKV6Attention"""
    def __init__(self, d_model=None, num_heads=8, chunk_size=64, hidden_size=None, **kwargs):
        super().__init__()
        # Handle both d_model and hidden_size naming
        self.d_model = d_model or hidden_size
        self.num_heads = num_heads
        self.head_dim = self.d_model // num_heads
        self.chunk_size = chunk_size
        
        # RWKV6-style parameters
        self.time_decay = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.time_first = nn.Parameter(torch.randn(num_heads, self.head_dim))
        
        # Projections
        self.receptance = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value = nn.Linear(self.d_model, self.d_model, bias=False)
        self.gate = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Channel mixing
        self.channel_key = nn.Linear(self.d_model, self.d_model * 4, bias=False)
        self.channel_value = nn.Linear(self.d_model * 4, self.d_model, bias=False)
        self.channel_receptance = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Normalization
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        
    def forward(self, x, **kwargs):
        B, T, C = x.shape
        
        # Time mixing
        x_ln = self.ln1(x)
        r = self.receptance(x_ln)
        k = self.key(x_ln)
        v = self.value(x_ln)
        g = F.silu(self.gate(x_ln))
        
        # Simplified WKV
        wkv = torch.softmax(k, dim=-1) * v
        out = self.output(r * wkv) * g
        x = x + out
        
        # Channel mixing
        x_ln = self.ln2(x)
        k = self.channel_key(x_ln)
        v = F.relu(k)**2
        v = self.channel_value(v)
        r = torch.sigmoid(self.channel_receptance(x_ln))
        x = x + v * r
        
        # Return tuple like FLA does: (output, None, None)
        return (x, None, None)

# ========== Mamba2 Replacement ==========
class Mamba2(nn.Module):
    """Replacement for fla.layers.mamba2.Mamba2"""
    def __init__(self, d_model=None, d_state=16, d_conv=4, expand=2, hidden_size=None, **kwargs):
        super().__init__()
        
        # Handle both d_model and hidden_size naming
        d_model = d_model or hidden_size
        
        # Try to use real mamba-ssm if available
        try:
            from mamba_ssm import Mamba as RealMamba
            self.real_mamba = RealMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_real_mamba = True
        except ImportError as e:
            print(f"Could not import real Mamba: {e}")
            self.use_real_mamba = False
            self.d_model = d_model
            self.d_state = d_state
            self.d_inner = d_model * expand
            
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  groups=self.d_inner, padding=d_conv - 1)
            self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x, **kwargs):
        if self.use_real_mamba:
            return self.real_mamba(x)
        else:
            B, T, C = x.shape
            x_proj = self.in_proj(x)
            x, gate = x_proj.chunk(2, dim=-1)
            x = x.transpose(1, 2)
            x = self.conv(x)[:, :, :T]
            x = x.transpose(1, 2)
            x = F.silu(x)
            x = x * self.D.unsqueeze(0).unsqueeze(0)
            x = x * F.silu(gate)
            x = self.out_proj(x)
            return x

print("FLA replacements loaded!")
