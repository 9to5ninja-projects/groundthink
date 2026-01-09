"""
Diagnose the RWKV/Mamba balance in GroundThink hybrid.

Key questions:
1. How much does base_decay (RWKV grounding) vs selective_decay (Mamba thinking) contribute?
2. What are the learned decay values? Are we remembering or forgetting?
3. Is the grounding mechanism actually affecting the output?
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from pathlib import Path
import math

from layers import GroundThinkBlock, RMSNorm, GroundingMechanism


def analyze_model_balance(model_path='groundthink_hybrid_v2.pt'):
    """Load trained model and analyze component contributions"""
    
    print("=" * 60)
    print("GroundThink Hybrid: RWKV/Mamba Balance Analysis")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']
    config = checkpoint.get('config', {})
    
    print(f"\nModel config: {config}")
    
    # Analyze each block
    print("\n" + "=" * 60)
    print("Per-Block Analysis")
    print("=" * 60)
    
    for i in range(6):  # Assuming 6 layers
        prefix = f'blocks.{i}.time_mixing'
        
        # Get grounding parameters
        base_decay = state_dict.get(f'{prefix}.grounding.base_decay')
        res_weight = state_dict.get(f'{prefix}.grounding.res_weight')
        time_decay_weight = state_dict.get(f'{prefix}.time_decay.weight')
        gate_weight = state_dict.get(f'{prefix}.gate.weight')
        
        print(f"\n--- Block {i} ---")
        
        if base_decay is not None:
            # base_decay -> exp(-base_decay) gives the RWKV retention factor
            retention = torch.exp(-base_decay)
            print(f"  RWKV Base Retention (grounding):")
            print(f"    mean: {retention.mean().item():.4f}")
            print(f"    min:  {retention.min().item():.4f}")
            print(f"    max:  {retention.max().item():.4f}")
            print(f"    std:  {retention.std().item():.4f}")
            
            # Interpret: 0.9 = remember 90%, 0.1 = forget 90%
            forget_pct = (1 - retention.mean().item()) * 100
            print(f"    → Avg {forget_pct:.1f}% forgetting per step (grounding floor)")
        
        if res_weight is not None:
            print(f"  Residual Weight: {res_weight.item():.4f}")
        
        if time_decay_weight is not None:
            # This is the projection that creates input-dependent decay
            # Larger norms = more dynamic/selective behavior
            norm = time_decay_weight.norm().item()
            print(f"  Mamba-like time_decay projection norm: {norm:.4f}")
            
            # Estimate typical output range
            # The output goes through exp(-exp(w)), so:
            # w ~ 0 → decay ~ 0.37 (moderate)
            # w ~ -1 → decay ~ 0.69 (slow)
            # w ~ 1 → decay ~ 0.07 (fast)
            
        if gate_weight is not None:
            # Gate controls how much of the state-based output is used
            norm = gate_weight.norm().item()
            print(f"  Output gate projection norm: {norm:.4f}")
    
    # Compute overall balance
    print("\n" + "=" * 60)
    print("Overall Balance Summary")
    print("=" * 60)
    
    # Collect all base_decay values
    all_retention = []
    all_time_decay_norms = []
    
    for i in range(6):
        prefix = f'blocks.{i}.time_mixing'
        base_decay = state_dict.get(f'{prefix}.grounding.base_decay')
        time_decay_weight = state_dict.get(f'{prefix}.time_decay.weight')
        
        if base_decay is not None:
            retention = torch.exp(-base_decay)
            all_retention.append(retention.mean().item())
        
        if time_decay_weight is not None:
            all_time_decay_norms.append(time_decay_weight.norm().item())
    
    if all_retention:
        avg_retention = sum(all_retention) / len(all_retention)
        print(f"\nRWKV Grounding (base_decay):")
        print(f"  Average retention across layers: {avg_retention:.4f}")
        print(f"  → Model remembers ~{avg_retention*100:.1f}% per step from grounding alone")
        
        if avg_retention > 0.8:
            print(f"  ⚡ GROUNDING DOMINANT: Model relies heavily on fixed memory")
        elif avg_retention < 0.3:
            print(f"  ⚡ SELECTIVE DOMINANT: Model forgets aggressively, relies on input selection")
        else:
            print(f"  ⚡ BALANCED: Hybrid behavior, both mechanisms active")
    
    if all_time_decay_norms:
        avg_norm = sum(all_time_decay_norms) / len(all_time_decay_norms)
        print(f"\nMamba Selectivity (time_decay projection):")
        print(f"  Average projection norm: {avg_norm:.4f}")
        
        if avg_norm > 5:
            print(f"  ⚡ HIGH SELECTIVITY: Input strongly modulates decay")
        elif avg_norm < 1:
            print(f"  ⚡ LOW SELECTIVITY: Decay is mostly fixed, not input-dependent")
        else:
            print(f"  ⚡ MODERATE: Some input-dependent behavior")


def test_live_balance(model_path='groundthink_hybrid_v2.pt'):
    """Run inference and measure actual decay values"""
    
    print("\n" + "=" * 60)
    print("Live Inference: Measuring Actual Decay Values")
    print("=" * 60)
    
    # Rebuild model
    from train_hybrid_v2 import GroundThinkModel, CharTokenizer
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    vocab = checkpoint['vocab']
    config = checkpoint['config']
    
    # Create tokenizer from vocab
    class LoadedTokenizer:
        def __init__(self, vocab):
            self.char_to_id = vocab
            self.id_to_char = {i: c for c, i in vocab.items()}
            self.vocab_size = len(vocab)
        def encode(self, text):
            return [self.char_to_id.get(c, 0) for c in text]
        def decode(self, ids):
            return ''.join(self.id_to_char.get(i, '?') for i in ids)
    
    tokenizer = LoadedTokenizer(vocab)
    
    model = GroundThinkModel(
        vocab_size=tokenizer.vocab_size,
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        head_dim=config['head_dim']
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test prompts
    prompts = [
        "Once upon a time there was",
        "The quick brown fox jumps",
        "I think therefore I am",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        tokens = torch.tensor([tokenizer.encode(prompt)])
        
        # Hook to capture decay values
        decay_values = []
        
        def capture_decay(module, input, output):
            if isinstance(module, GroundingMechanism):
                grounded_x, grounded_decay = output
                decay_values.append(grounded_decay.detach())
        
        hooks = []
        for block in model.blocks:
            if hasattr(block.time_mixing, 'grounding'):
                hooks.append(block.time_mixing.grounding.register_forward_hook(capture_decay))
        
        with torch.no_grad():
            _ = model(tokens)
        
        for h in hooks:
            h.remove()
        
        # Analyze captured decays
        for i, decay in enumerate(decay_values):
            # decay shape: [B, L, D]
            mean_decay = decay.mean().item()
            min_decay = decay.min().item()
            max_decay = decay.max().item()
            std_decay = decay.std().item()
            
            print(f"  Block {i}: decay mean={mean_decay:.4f} range=[{min_decay:.4f}, {max_decay:.4f}] std={std_decay:.4f}")


if __name__ == "__main__":
    model_path = 'groundthink_hybrid_v2.pt'
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train first with train_hybrid_v2.py")
    else:
        analyze_model_balance(model_path)
        test_live_balance(model_path)
