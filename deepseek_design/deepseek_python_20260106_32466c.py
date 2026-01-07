class GroundingMechanism(nn.Module):
    """Ensures state doesn't drift - RWKV's contribution"""
    def __init__(self, dim):
        super().__init__()
        # Three-tier grounding:
        # 1. Local (conv)
        self.conv_short = nn.Conv1d(dim, dim, kernel_size=4, groups=dim)
        
        # 2. Medium-term (learned base decay)
        self.base_decay = nn.Parameter(torch.ones(dim))
        
        # 3. Long-term (residual pathway)
        self.res_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x, state, dt):
        # Base decay provides stability floor
        w_base = torch.exp(-self.base_decay.view(1, 1, -1))
        
        # Selective decay from Mamba provides flexibility
        w_selective = torch.exp(-dt)
        
        # Combined: grounding + thinking
        w_combined = w_base * w_selective
        
        # Ensure minimum retention (never completely forget)
        w_combined = torch.clamp(w_combined, min=0.01, max=0.99)
        
        return w_combined