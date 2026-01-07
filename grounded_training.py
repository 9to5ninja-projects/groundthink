"""
GroundedMamba Training Utilities
Per FOUNDATION.md specification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(model, lr: float = 3e-4):
    """
    Create optimizer with separate param groups per FOUNDATION.md:
    - SSM params: lr * 0.5, weight_decay=0.1
    - RWKV params: lr * 0.3, weight_decay=0.01
    - Norm params: lr * 1.0, weight_decay=0.0
    - Other params: lr * 1.0, weight_decay=0.1
    """
    param_groups = model.get_param_groups(lr)
    return AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)


def get_scheduler(optimizer, warmup_steps: int = 1000, total_steps: int = 50000):
    """
    Warmup + cosine decay scheduler per FOUNDATION.md
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def compute_loss(logits, targets, states, stability_weight: float = 0.1):
    """
    Loss with stability penalties per FOUNDATION.md:
    - Cross-entropy
    - State stability penalty (norm towards 5.0)
    - Output consistency penalty
    """
    # 1. Cross-entropy
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100
    )
    
    # 2. State stability penalty
    stability_loss = torch.tensor(0.0, device=logits.device)
    for layer_state in states:
        if layer_state is None:
            continue
        for key, state in layer_state.items():
            if state is not None:
                state_norm = torch.norm(state, dim=-1).mean()
                target_norm = 5.0
                stability_loss = stability_loss + F.mse_loss(
                    state_norm,
                    torch.tensor(target_norm, device=state.device)
                )
    
    # 3. Output consistency penalty
    if logits.size(1) > 1:
        output_diff = logits[:, 1:] - logits[:, :-1]
        consistency_loss = output_diff.norm(dim=-1).mean() * 0.01
    else:
        consistency_loss = torch.tensor(0.0, device=logits.device)
    
    total_loss = ce_loss + stability_weight * stability_loss + consistency_loss
    
    return total_loss, {
        'ce_loss': ce_loss.item(),
        'stability_loss': stability_loss.item(),
        'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
        'total_loss': total_loss.item(),
    }


class GradientDebugger:
    """
    Tracks gradient issues in real-time.
    Per FOUNDATION.md debugging procedures.
    """
    
    def __init__(self, model, log_interval: int = 100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        self.nan_count = 0
        self.inf_count = 0
        
    def check_and_fix(self):
        """Check for NaN/Inf gradients, replace with zeros"""
        self.step_count += 1
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            nan_mask = torch.isnan(param.grad)
            inf_mask = torch.isinf(param.grad)
            
            if nan_mask.any():
                self.nan_count += 1
                param.grad = torch.where(nan_mask, torch.zeros_like(param.grad), param.grad)
                print(f"⚠️ NaN gradient in {name} at step {self.step_count}")
            
            if inf_mask.any():
                self.inf_count += 1
                param.grad = torch.where(inf_mask, torch.zeros_like(param.grad), param.grad)
                print(f"⚠️ Inf gradient in {name} at step {self.step_count}")
        
        return self.nan_count, self.inf_count


class StateMonitor:
    """
    Monitors state norms for stability.
    Per FOUNDATION.md: healthy range is 0.1 - 10.0
    """
    
    def __init__(self, warn_threshold: float = 10.0, error_threshold: float = 100.0):
        self.warn_threshold = warn_threshold
        self.error_threshold = error_threshold
        
    def check(self, states):
        """Check state norms, return stats"""
        stats = {}
        warnings = []
        
        for i, layer_state in enumerate(states):
            if layer_state is None:
                continue
            for key, state in layer_state.items():
                if state is None:
                    continue
                norm = torch.norm(state, dim=-1).mean().item()
                name = f"layer_{i}_{key}_norm"
                stats[name] = norm
                
                if norm > self.error_threshold:
                    warnings.append(f"❌ {name}: {norm:.2f} (> {self.error_threshold})")
                elif norm > self.warn_threshold:
                    warnings.append(f"⚠️ {name}: {norm:.2f} (> {self.warn_threshold})")
        
        return stats, warnings


class EmergencyRecovery:
    """
    Auto-rollback when loss explodes.
    Per FOUNDATION.md: keep golden checkpoints.
    """
    
    def __init__(self, model, optimizer, max_buffer: int = 5):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_buffer = []
        self.max_buffer = max_buffer
        
    def save_good_state(self, step: int):
        """Save a known-good state"""
        state = {
            'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer': self.optimizer.state_dict(),
            'step': step,
        }
        self.checkpoint_buffer.append(state)
        if len(self.checkpoint_buffer) > self.max_buffer:
            self.checkpoint_buffer.pop(0)
    
    def recover(self):
        """Roll back to last good state, reduce LR by 0.7"""
        if not self.checkpoint_buffer:
            print("❌ No good states to recover from!")
            return False
        
        good_state = self.checkpoint_buffer[-1]
        self.model.load_state_dict(good_state['model'])
        self.optimizer.load_state_dict(good_state['optimizer'])
        
        # Reduce LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.7
        
        new_lr = self.optimizer.param_groups[0]['lr']
        print(f"🔥 RECOVERY: Rolled back to step {good_state['step']}, LR -> {new_lr:.6f}")
        return True


def check_loss_health(loss, optimizer):
    """
    Check for NaN/Inf loss, reduce LR if detected.
    Per FOUNDATION.md debugging procedures.
    """
    if torch.isnan(loss) or torch.isinf(loss):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        new_lr = optimizer.param_groups[0]['lr']
        print(f"❌ NaN/Inf loss! LR reduced to {new_lr:.6f}")
        return False
    return True

