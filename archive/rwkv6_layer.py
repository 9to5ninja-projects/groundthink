"""
Minimal RWKV6 implementation without FLA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRWKV6Layer(nn.Module):
    """Simple RWKV6-like layer for testing"""
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        self.dim = dim
        self.expansion = expansion_factor
        
        # Time mixing components
        self.time_decay = nn.Parameter(torch.randn(dim))
        self.time_first = nn.Parameter(torch.randn(dim))
        
        # Projections
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        
        # Channel mixing
        self.channel_key = nn.Linear(dim, dim * expansion_factor, bias=False)
        self.channel_value = nn.Linear(dim * expansion_factor, dim, bias=False)
        self.channel_receptance = nn.Linear(dim, dim, bias=False)
        self.channel_gate = nn.Linear(dim, dim, bias=False)
        
        # Normalization
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Time mixing (simplified)
        B, T, C = x.shape
        
        x_ln = self.ln1(x)
        r = self.receptance(x_ln)
        k = self.key(x_ln)
        v = self.value(x_ln)
        g = F.silu(self.gate(x_ln))
        
        # Simplified WKV computation
        # In real RWKV6, this is more complex with chunk processing
        wkv = torch.softmax(k, dim=-1) * v
        x = x + self.output(r * wkv) * g
        
        # Channel mixing
        x_ln = self.ln2(x)
        k = self.channel_key(x_ln)
        v = F.relu(k)**2
        v = self.channel_value(v)
        r = torch.sigmoid(self.channel_receptance(x_ln))
        g = torch.sigmoid(self.channel_gate(x_ln))
        x = x + v * r * g
        
        return x

# Test
if __name__ == "__main__":
    layer = SimpleRWKV6Layer(128).cuda()
    x = torch.randn(2, 32, 128).cuda()
    y = layer(x)
    print(f"RWKV6 Layer test: {x.shape} -> {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in layer.parameters()):,}")
