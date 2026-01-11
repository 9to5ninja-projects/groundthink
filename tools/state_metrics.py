"""
State Metrics Tracker (Task 53)

Lightweight utility to track state health during training.
Captures RWKV/Mamba norms, variance, and component ratios over time.

Usage:
    from tools.state_metrics import StateMetrics
    
    tracker = StateMetrics()
    
    # During training loop:
    _, states = model(x, return_states=True)
    tracker.log(step, states)
    
    # After training:
    tracker.summary()
    tracker.save('logs/state_metrics.json')

Reference: VALIDATION_ROADMAP.md, Task 53
Created: 2026-01-10
"""

import json
from pathlib import Path
from typing import Dict, Optional


class StateMetrics:
    """Track state evolution during training."""
    
    def __init__(self):
        self.history = {
            'step': [],
            'rwkv_norm': [],
            'mamba_norm': [],
            'rwkv_var': [],
            'mamba_var': [],
            'ratio': [],  # rwkv_norm / mamba_norm
        }
        self._last_step = -1
    
    def log(self, step: int, states: Dict):
        """
        Log state metrics at a training step.
        
        Args:
            step: Current training step
            states: Dict with 'rwkv_state' and 'mamba_state' tensors
        """
        # Avoid duplicate logging
        if step == self._last_step:
            return
        self._last_step = step
        
        rwkv_state = states.get('rwkv_state')
        mamba_state = states.get('mamba_state')
        
        if rwkv_state is None or mamba_state is None:
            return
        
        rwkv_norm = rwkv_state.norm().item()
        mamba_norm = mamba_state.norm().item()
        rwkv_var = rwkv_state.var().item()
        mamba_var = mamba_state.var().item()
        ratio = rwkv_norm / (mamba_norm + 1e-8)
        
        self.history['step'].append(step)
        self.history['rwkv_norm'].append(rwkv_norm)
        self.history['mamba_norm'].append(mamba_norm)
        self.history['rwkv_var'].append(rwkv_var)
        self.history['mamba_var'].append(mamba_var)
        self.history['ratio'].append(ratio)
    
    def summary(self) -> Dict:
        """Print and return summary statistics."""
        if not self.history['step']:
            print("No data logged yet.")
            return {}
        
        n = len(self.history['step'])
        first_step = self.history['step'][0]
        last_step = self.history['step'][-1]
        
        # Compute stats
        stats = {
            'n_samples': n,
            'step_range': (first_step, last_step),
            'rwkv_norm_mean': sum(self.history['rwkv_norm']) / n,
            'mamba_norm_mean': sum(self.history['mamba_norm']) / n,
            'ratio_mean': sum(self.history['ratio']) / n,
            'ratio_first': self.history['ratio'][0] if n > 0 else None,
            'ratio_last': self.history['ratio'][-1] if n > 0 else None,
        }
        
        print(f"\n📊 State Metrics Summary (steps {first_step}-{last_step})")
        print(f"   Samples: {n}")
        print(f"   RWKV norm (mean): {stats['rwkv_norm_mean']:.2f}")
        print(f"   Mamba norm (mean): {stats['mamba_norm_mean']:.4f}")
        print(f"   Ratio (mean): {stats['ratio_mean']:.1f}x")
        print(f"   Ratio drift: {stats['ratio_first']:.1f}x → {stats['ratio_last']:.1f}x")
        
        return stats
    
    def save(self, path: str):
        """Save history to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✓ Saved state metrics to {path}")
    
    def load(self, path: str):
        """Load history from JSON file."""
        with open(path, 'r') as f:
            self.history = json.load(f)
        
        self._last_step = self.history['step'][-1] if self.history['step'] else -1
        print(f"✓ Loaded {len(self.history['step'])} samples from {path}")


# Quick test
if __name__ == '__main__':
    import torch
    
    # Simulate state data
    tracker = StateMetrics()
    
    for step in range(0, 100, 10):
        fake_states = {
            'rwkv_state': torch.randn(1, 4, 32) * 100,
            'mamba_state': torch.randn(1, 128) * 0.5,
        }
        tracker.log(step, fake_states)
    
    tracker.summary()
