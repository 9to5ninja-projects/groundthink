"""
Modified RWKV-6 Model for Base Model Characterization (Task 0.0.1)

GroundThink — Phase 0: Base Model Characterization
Copyright (c) 2026 9to5ninja

IMPORTANT: This is NOT pure RWKV-6 from the original paper.
We use GroundThink's custom ops/ package instead of the Flash Linear Attention (FLA)
library due to dependency management issues. The effects of this modification are UNKNOWN
and accepted as baseline reality for Phase 0.

Future work may explore FLA integration for comparative testing, but for progress we proceed
with this modified implementation as our "pure" RWKV-6 baseline.

ATTRIBUTION:
This implementation uses RWKV-6 architecture from:
    - Paper: "Eagle and Finch: RWKV with Matrix-Valued States" (Peng et al., 2024)
    - Implementation: GroundThink ops/ package (custom CUDA wrapper + PyTorch prototype)
                      NOT the FLA library from the original paper
    - Original: https://github.com/BlinkDL/RWKV-LM

OUR CONTRIBUTION:
    - Modified RWKV-6 for baseline characterization (no fusion, no FLA)
    - 4M parameter configuration for fair comparison with hybrid models
    - Integration with GroundThink validation methodology

If you use this code, please cite:
    1. RWKV-6 paper (Peng et al., 2024)
    2. GroundThink project (this work)
    3. Note the FLA modification in your methodology
See ATTRIBUTION.md for full citation details.

Architecture:
- Embedding layer (tied with output head)
- N x RWKV-6 blocks (RMSNorm + RWKV6Attention + RMSNorm + FFN)
- Output head (shares weights with embeddings)

Target: ~4M parameters for direct comparison with hybrid models.
"""

import torch
import torch.nn as nn
from ops import RWKV6Attention  # Uses GroundThink ops/ package, not FLA


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RWKV6Block(nn.Module):
    """
    Pure RWKV-6 block consisting of:
    - Pre-norm + RWKV6Attention
    - Pre-norm + FFN
    Both with residual connections.
    """
    def __init__(self, hidden_size: int, ffn_mult: float = 4.0, num_heads: int = 4, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Pre-norm layers (RMSNorm standard for modern LLMs)
        self.ln1 = RMSNorm(hidden_size)
        
        # RWKV-6 attention mechanism (using ops/ package)
        self.rwkv = RWKV6Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_idx=layer_idx,
        )
        
        self.ln2 = RMSNorm(hidden_size)
        ffn_hidden = int(hidden_size * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden, bias=False),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size, bias=False),
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            x: [batch_size, seq_len, hidden_size]
        """
        # RWKV attention pathway (ops.RWKV6Attention returns (output, None, state_dict))
        x = x + self.rwkv(self.ln1(x))[0]
        
        # FFN pathway
        x = x + self.ffn(self.ln2(x))
        
        return x


class RWKV6Model(nn.Module):
    """
    Pure RWKV-6 language model for Task 0.0.1 characterization.
    
    Configuration:
    - 8 layers × 144 hidden (tied embeddings) ≈ 4.0M parameters
    - Designed to match hybrid model scale for fair comparison
    """
    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_size: int = 144,
        num_layers: int = 8,
        ffn_mult: float = 4.0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings
        
        # Input embedding
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # RWKV-6 blocks
        self.blocks = nn.ModuleList([
            RWKV6Block(hidden_size, ffn_mult, num_heads=4, layer_idx=i)
            for i in range(num_layers)
        ])
        
        # Output projection
        self.ln_out = RMSNorm(hidden_size)
        if tie_embeddings:
            self.head = None  # Will use embed.weight transposed
        else:
            self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all parameters"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Token IDs [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        h = self.embed(x)  # [B, L, H]
        
        # Pass through all RWKV-6 blocks
        for block in self.blocks:
            h = block(h)
        
        # Final norm and project to vocab
        h = self.ln_out(h)
        
        if self.tie_embeddings:
            # Manual tied projection: h @ embed.weight.T
            logits = torch.matmul(h, self.embed.weight.T)  # [B, L, V]
        else:
            logits = self.head(h)
        
        return logits
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters, optionally excluding embeddings"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
        return n_params


def create_rwkv6_4m(vocab_size: int = 16000) -> RWKV6Model:
    """
    Create ~4M parameter pure RWKV-6 model for Task 0.0.1.
    
    Configuration chosen to match hybrid model parameter count
    for fair comparison during base model characterization.
    """
    return RWKV6Model(
        vocab_size=vocab_size,
        hidden_size=144,
        num_layers=8,
        ffn_mult=4.0,
        tie_embeddings=True,
    )


# Quick parameter count verification
if __name__ == "__main__":
    print("=== RWKV-6 Pure Model Parameter Check ===\n")
    
    model = create_rwkv6_4m(vocab_size=16000)
    
    total_params = model.get_num_params(non_embedding=False)
    non_embed_params = model.get_num_params(non_embedding=True)
    
    print(f"Vocab size: {model.vocab_size}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Num layers: {model.num_layers}")
    print(f"Tied embeddings: {model.tie_embeddings}")
    print()
    print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Non-embedding params: {non_embed_params:,} ({non_embed_params/1e6:.2f}M)")
    print()
    
    if 3.5e6 <= total_params <= 4.5e6:
        print("✓ Parameter count within target range (3.5-4.5M)")
    else:
        print("✗ Parameter count outside target range")
    
    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    x = torch.randint(0, 16000, (2, 64), device=device)  # batch=2, seq=64
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 64, 16000), "Shape mismatch!"
    print("✓ Forward pass successful")
