"""
Training utilities for GroundThink

Based on DeepSeek recommendations:
- Gradient clipping for SSM stability
- Cosine LR with warmup
- State normalization
- Curriculum learning on sequence length
"""

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Iterator


def clip_grad_norm_ssm(parameters: Iterator[nn.Parameter], max_norm: float = 1.0) -> float:
    """
    Special gradient clipping for SSM parameters.
    More aggressive than standard clipping to prevent state explosion.
    """
    parameters = list(parameters)
    total_norm = 0.0
    
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return total_norm


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 2000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine decay with linear warmup.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of peak (0.1 = 10% of peak)
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)


def create_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> AdamW:
    """
    Create AdamW optimizer with weight decay only on non-bias/norm parameters.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay on biases, norms, and embeddings
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    return AdamW(optim_groups, lr=lr, betas=betas)


class CurriculumScheduler:
    """
    Curriculum learning for sequence length.
    
    Start with short sequences (4K), gradually increase to target (128K).
    This helps the model learn stable state dynamics before long-context.
    """
    
    def __init__(
        self,
        initial_seq_len: int = 4096,
        target_seq_len: int = 131072,
        warmup_steps: int = 10000,
        schedule: str = "linear",  # or "exponential"
    ):
        self.initial = initial_seq_len
        self.target = target_seq_len
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.current_step = 0
    
    def step(self) -> int:
        """Get current sequence length and increment step"""
        if self.current_step >= self.warmup_steps:
            seq_len = self.target
        elif self.schedule == "linear":
            ratio = self.current_step / self.warmup_steps
            seq_len = int(self.initial + (self.target - self.initial) * ratio)
        else:  # exponential
            ratio = self.current_step / self.warmup_steps
            log_initial = math.log2(self.initial)
            log_target = math.log2(self.target)
            seq_len = int(2 ** (log_initial + (log_target - log_initial) * ratio))
        
        self.current_step += 1
        return seq_len
    
    def reset(self):
        self.current_step = 0


class GradientCheckpointer:
    """
    Wrapper for gradient checkpointing during training.
    Reduces memory at cost of recomputation.
    """
    
    def __init__(self, model: nn.Module, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        
        if enabled:
            for block in model.blocks:
                block.time_mixing.use_checkpoint = True
    
    def __call__(self, func, *args):
        if self.enabled:
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)
        return func(*args)


class TrainingState:
    """Track training metrics"""
    
    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.losses = []
        self.grad_norms = []
    
    def update(self, loss: float, grad_norm: float):
        self.step += 1
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
        
        if loss < self.best_loss:
            self.best_loss = loss
    
    def get_avg_loss(self, window: int = 100) -> float:
        if not self.losses:
            return 0.0
        return sum(self.losses[-window:]) / len(self.losses[-window:])
    
    def get_avg_grad_norm(self, window: int = 100) -> float:
        if not self.grad_norms:
            return 0.0
        return sum(self.grad_norms[-window:]) / len(self.grad_norms[-window:])
