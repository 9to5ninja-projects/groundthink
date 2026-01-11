"""
GPT-2 Baseline Model for Controlled Comparison

This is a minimal GPT-2 implementation designed for fair comparison
with GF-MH (Gated-Fusion Mamba-Heavy) hybrid architecture.

Experiment ID: EXP-001
Target: Match GF-MH param count (5,618,920)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, d_model: int, n_heads: int, max_pos: int = 1024, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Combined QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (will be moved to device on first forward)
        self.register_buffer("bias", torch.tril(torch.ones(max_pos, max_pos)).view(1, 1, max_pos, max_pos))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out)


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_pos: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_pos, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    """
    Minimal GPT-2 architecture for baseline comparison.
    
    Architecture:
    - Token embedding + positional embedding
    - N transformer blocks (pre-norm)
    - Final layer norm
    - LM head (tied with token embeddings)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        max_pos: int = 1024,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff or (4 * d_model)
        self.max_pos = max_pos
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, self.d_ff, max_pos, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            targets: Target token IDs for loss computation
            
        Returns:
            logits: LM logits of shape (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        B, T = input_ids.shape
        assert T <= self.max_pos, f"Sequence length {T} exceeds max_pos {self.max_pos}"
        
        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        
        # Embeddings
        tok_emb = self.token_emb(input_ids)  # (B, T, d_model)
        pos_emb = self.pos_emb(pos)  # (T, d_model)
        x = self.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
        
        return logits, loss
    
    def get_num_params(self, exclude_embeddings: bool = False) -> int:
        """Get total parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n_params -= self.token_emb.weight.numel()
            n_params -= self.pos_emb.weight.numel()
        return n_params


# ============================================================================
# Factory functions for specific sizes
# ============================================================================

def create_gpt2_5m(vocab_size: int = 10000) -> GPT2Model:
    """
    GPT-2 baseline ~5.5M params (close to GF-MH's 5.6M).
    
    Config: 8 layers, d_model=192, 4 heads
    Standard transformer for baseline comparison.
    """
    return GPT2Model(
        vocab_size=vocab_size,
        d_model=192,
        n_layers=8,
        n_heads=4,  # 192 / 4 = 48 per head
        d_ff=768,   # 4 * 192
        max_pos=1024,
        dropout=0.0,
        tie_weights=True,
    )


def create_gpt2_3m(vocab_size: int = 10000) -> GPT2Model:
    """Smaller GPT-2 for quick tests."""
    return GPT2Model(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_pos=1024,
        dropout=0.0,
        tie_weights=True,
    )


def create_gpt2_8m(vocab_size: int = 10000) -> GPT2Model:
    """Larger GPT-2 for scaling tests."""
    return GPT2Model(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=10,
        n_heads=4,
        d_ff=1024,
        max_pos=1024,
        dropout=0.0,
        tie_weights=True,
    )


# ============================================================================
# Quick validation
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("GPT-2 Baseline Model Validation")
    print("=" * 60)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = create_gpt2_5m(vocab_size=10000)
    model = model.to(device)
    
    # Count parameters
    total_params = model.get_num_params()
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Target (GF-MH): 5,618,920")
    print(f"Difference: {total_params - 5618920:+,} ({(total_params/5618920 - 1)*100:+.3f}%)")
    
    # Forward pass test
    batch_size = 4
    seq_len = 64
    x = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    
    print(f"\nForward pass test:")
    print(f"  Input shape: {x.shape}")
    
    with torch.no_grad():
        logits, loss = model(x, targets)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Memory footprint
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {mem:.1f} MB")
    
    print("\n✓ GPT-2 baseline ready for comparison")
    print("=" * 60)
