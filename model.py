"""
GroundThink Full Model

Combines RWKV-7's selective state mechanism with grounding layers
to create a model that can both "think" dynamically and stay "grounded"
in long contexts without drift.
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import GroundThinkConfig, GROUNDTHINK_160M
from .layers import GroundThinkBlock, RMSNorm


class GroundThinkModel(nn.Module):
    """
    GroundThink: RWKV-7 with Enhanced Grounding
    
    Architecture:
    - Embedding layer
    - N x GroundThinkBlock (TimeMixing + ChannelMixing with Grounding)
    - Output projection
    
    Key innovation: GroundingMechanism in each TimeMixing layer
    prevents state drift while preserving RWKV-7's selective learning.
    """
    
    def __init__(self, config: GroundThinkConfig = GROUNDTHINK_160M):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            GroundThinkBlock(
                dim=config.d_model,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                state_dim=config.state_dim,
                ffn_expansion=config.ffn_expansion,
                use_grounding=config.use_grounding,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_out = RMSNorm(config.d_model)
        
        # Output projection (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
        
        # Initialize with power-law decay across layers
        self._init_decay_schedule()
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def _init_decay_schedule(self):
        """
        Initialize decay parameters with power-law schedule:
        - Early layers: faster decay (local focus)
        - Later layers: slower decay (global focus)
        """
        if self.config.w_init != "power_law":
            return
            
        for i, block in enumerate(self.blocks):
            # Ratio: 0 for first layer, 1 for last layer
            ratio = i / max(1, self.config.n_layers - 1)
            
            # Early layers decay faster (smaller retention)
            # Later layers decay slower (larger retention)
            base_retention = 0.9 + 0.09 * ratio  # 0.9 to 0.99
            
            if hasattr(block.time_mixing, 'grounding'):
                with torch.no_grad():
                    # Set base_decay to achieve target retention
                    # retention = exp(-base_decay), so base_decay = -log(retention)
                    target_decay = -torch.log(torch.tensor(base_retention))
                    block.time_mixing.grounding.base_decay.fill_(target_decay.item())
    
    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[list[torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """
        Args:
            input_ids: (B, L) token indices
            states: List of previous states, one per layer
            return_states: Whether to return updated states
        
        Returns:
            logits: (B, L, vocab_size)
            new_states: Updated states if return_states=True
        """
        B, L = input_ids.shape
        
        # Initialize states if needed
        if states is None:
            states = [None] * self.config.n_layers
        
        # Embed tokens
        x = self.embedding(input_ids)
        
        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, state = block(x, states[i])
            new_states.append(state)
        
        # Final norm and projection
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        if return_states:
            return logits, new_states
        return logits, None
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple greedy/sampling generation"""
        self.eval()
        states = None
        generated = input_ids.clone()
        
        with torch.no_grad():
            # Process prompt
            logits, states = self.forward(input_ids, states, return_states=True)
            
            for _ in range(max_new_tokens):
                # Get logits for last token
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append and continue
                generated = torch.cat([generated, next_token], dim=1)
                
                # Forward single token
                logits, states = self.forward(next_token, states, return_states=True)
        
        return generated
    
    @classmethod
    def from_rwkv7(cls, rwkv7_path: str, config: Optional[GroundThinkConfig] = None):
        """
        Initialize GroundThink from pretrained RWKV-7 weights.
        Adds grounding layers on top of existing RWKV-7 structure.
        """
        # Load RWKV-7 weights
        state_dict = torch.load(rwkv7_path, weights_only=False)
        
        # Infer config from weights if not provided
        if config is None:
            # TODO: Infer dimensions from weight shapes
            raise NotImplementedError("Auto config inference not yet implemented")
        
        # Create model
        model = cls(config)
        
        # Map RWKV-7 weights to GroundThink structure
        # TODO: Implement weight mapping
        # The mapping would transfer:
        # - emb.weight -> embedding.weight
        # - blocks.N.att.* -> blocks.N.time_mixing.*
        # - blocks.N.ffn.* -> blocks.N.channel_mixing.*
        # - head.weight -> lm_head.weight
        
        raise NotImplementedError("RWKV-7 weight loading not yet implemented")
        
        return model


def count_parameters(model: nn.Module) -> dict:
    """Count parameters by component"""
    counts = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0]
        if component not in counts:
            counts[component] = 0
        counts[component] += param.numel()
    
    counts['total'] = sum(counts.values())
    return counts
