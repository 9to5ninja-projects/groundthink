class ProductionHybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use different precision for different components
        self.rwkv_layers = nn.ModuleList([
            SelectiveRWKVBlock().to(torch.bfloat16) for _ in range(config.n_layers//2)
        ])
        
        self.mamba_layers = nn.ModuleList([
            MambaBlock().to(torch.float32) for _ in range(config.n_layers//2)
        ])
        
        # Router for dynamic computation
        self.router = nn.Linear(config.d_model, 2)
        
    def forward(self, x):
        # Dynamic routing: when to use RWKV vs Mamba
        route_logits = self.router(x.mean(dim=1))
        route_weights = torch.softmax(route_logits, dim=-1)
        
        # Process through both, weight outputs
        rwkv_out = self._process_rwkv(x)
        mamba_out = self._process_mamba(x)
        
        # Weighted combination
        output = (route_weights[0] * rwkv_out + 
                 route_weights[1] * mamba_out)
        
        return output