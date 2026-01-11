"""
Gradient-State Coupling Analyzer (Task 54)

Measures correlation between gradient magnitudes and state changes.
Helps diagnose if gradients are actually updating state-related parameters.

Usage:
    from tools.gradient_coupling import GradientCoupling
    
    analyzer = GradientCoupling(model)
    
    # During training:
    loss.backward()
    analyzer.log(step)
    
    # After training:
    analyzer.summary()

Reference: VALIDATION_ROADMAP.md, Task 54
Created: 2026-01-10
"""

import torch
from typing import Dict, List, Optional


class GradientCoupling:
    """Analyze gradient flow to RWKV vs Mamba components."""
    
    def __init__(self, model):
        self.model = model
        self.history = {
            'step': [],
            'rwkv_grad_norm': [],
            'mamba_grad_norm': [],
            'other_grad_norm': [],
            'ratio': [],  # rwkv/mamba grad ratio
        }
        self._last_step = -1
    
    def _classify_params(self):
        """Classify parameters as RWKV, Mamba, or other."""
        rwkv_params = []
        mamba_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            name_lower = name.lower()
            if 'rwkv' in name_lower or 'wkv' in name_lower:
                rwkv_params.append(param)
            elif 'mamba' in name_lower or 'ssm' in name_lower:
                mamba_params.append(param)
            else:
                other_params.append(param)
        
        return rwkv_params, mamba_params, other_params
    
    def _compute_grad_norm(self, params: List) -> float:
        """Compute total gradient norm for a list of parameters."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def log(self, step: int):
        """Log gradient norms at current step (call after backward())."""
        if step == self._last_step:
            return
        self._last_step = step
        
        rwkv_params, mamba_params, other_params = self._classify_params()
        
        rwkv_norm = self._compute_grad_norm(rwkv_params)
        mamba_norm = self._compute_grad_norm(mamba_params)
        other_norm = self._compute_grad_norm(other_params)
        ratio = rwkv_norm / (mamba_norm + 1e-8)
        
        self.history['step'].append(step)
        self.history['rwkv_grad_norm'].append(rwkv_norm)
        self.history['mamba_grad_norm'].append(mamba_norm)
        self.history['other_grad_norm'].append(other_norm)
        self.history['ratio'].append(ratio)
    
    def summary(self) -> Dict:
        """Print and return summary statistics."""
        if not self.history['step']:
            print("No gradients logged yet.")
            return {}
        
        n = len(self.history['step'])
        
        stats = {
            'n_samples': n,
            'rwkv_grad_mean': sum(self.history['rwkv_grad_norm']) / n,
            'mamba_grad_mean': sum(self.history['mamba_grad_norm']) / n,
            'ratio_mean': sum(self.history['ratio']) / n,
        }
        
        print(f"\n📊 Gradient Coupling Summary ({n} samples)")
        print(f"   RWKV grad norm (mean): {stats['rwkv_grad_mean']:.6f}")
        print(f"   Mamba grad norm (mean): {stats['mamba_grad_mean']:.6f}")
        print(f"   Ratio (RWKV/Mamba): {stats['ratio_mean']:.2f}x")
        
        # Interpret
        if stats['ratio_mean'] < 0.1:
            print("   ⚠️  RWKV receiving much weaker gradients")
        elif stats['ratio_mean'] > 10:
            print("   ⚠️  Mamba receiving much weaker gradients")
        else:
            print("   ✓ Gradient flow balanced")
        
        return stats


# Quick test
if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from models import get_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model('GF-MH', vocab_size=16000)
    model = model.to(device)
    model.train()
    
    analyzer = GradientCoupling(model)
    
    # Simulate a forward/backward pass (seq_len must be multiple of 8)
    x = torch.randint(0, 16000, (1, 64), device=device)
    out = model(x)
    loss = out.mean()
    loss.backward()
    
    analyzer.log(step=0)
    analyzer.summary()
