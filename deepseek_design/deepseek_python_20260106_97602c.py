# 1. Gradient Clipping for SSMs
def clip_grad_norm_ssm(parameters, max_norm=1.0):
    """Special clipping for SSM parameters"""
    total_norm = 0
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

# 2. State Normalization (prevents explosion)
def normalize_state(state, eps=1e-5):
    """RMS normalization for state matrices"""
    norm = state.pow(2).mean(-1, keepdim=True).sqrt()
    return state / (norm + eps)

# 3. Learning Rate Schedule
def get_hybrid_schedule(warmup_steps=2000, total_steps=100000):
    """Cosine with warmup, adjusted for hybrid model"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda