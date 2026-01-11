"""
GroundThink V4 Training Script

Per V4_STRATEGY.md Task 4 requirements.
Updated 2026-01-10: Task 18.2 - YAML config + CLI overrides
"""

# Fix OpenMP conflict BEFORE any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import yaml
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pathlib import Path

from models import get_model, list_models
from data import load_stateful_dataset


# ============ Default Config ============
# Can be overridden by --config YAML file or CLI args
DEFAULT_CONFIG = {
    # Model (from registry)
    'model': '5M',
    
    # Optimizer
    'lr': 3e-4,
    'min_lr': 3e-5,           # 10% of peak
    'weight_decay': 0.1,
    'betas': (0.9, 0.95),
    'mamba_lr_mult': 0.5,     # Balance RWKV/Mamba gradients (1.0 tested, made worse)
    'mamba_grad_scale': 1.0,  # Gradient scaling for Mamba (1.0 = no scaling, >1 = boost Mamba grads)
    
    # RWKV Dropout (Force Mamba Learning)
    'rwkv_dropout_start': 0.0,  # Initial RWKV dropout prob (0.4 = drop 40% of time)
    'rwkv_dropout_end': 0.0,    # Final RWKV dropout prob (anneals linearly)
    
    # Schedule  
    'warmup_ratio': 0.1,      # 10% of max_steps (V3 guideline: 5-10%)
    'lr_decay': 'cosine',
    'lr_decay': 'cosine',
    
    # Batch
    'batch_size': 64,
    'seq_len': 64,
    'grad_accum': 1,
    'grad_clip': 1.0,
    
    # AMP
    'use_amp': True,
    
    # Training
    'max_steps': 5000,
    'eval_every': 100,
    'save_every': 1000,
    'log_every': 50,
    
    # Stopping
    'val_patience': 5,
    'grad_ratio_warn': 3.0,
    'grad_ratio_fail': 10.0,
    
    # Checkpoints
    'checkpoint_prefix': 'ckpt',
}


def load_config(config_path: str = None, cli_overrides: dict = None) -> dict:
    """
    Load config from YAML file with CLI overrides.
    
    Priority: CLI args > YAML file > DEFAULT_CONFIG
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load YAML if provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)
            print(f"Loaded config from {config_path}")
        else:
            print(f"Warning: Config file {config_path} not found, using defaults")
    
    # Apply CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                config[key] = value
    
    return config


# ============ Device ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============ Parameter Groups ============
def get_parameter_groups(model, base_lr, mamba_lr_mult=2.0, weight_decay=0.1):
    """
    Separate RWKV and Mamba parameters for differential LR.
    Per V4_DESIGN.md: Mamba may need 1.5-3x higher LR.
    """
    rwkv_params = []
    mamba_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'rwkv' in name.lower():
            rwkv_params.append(param)
        elif 'mamba' in name.lower():
            mamba_params.append(param)
        else:
            other_params.append(param)
    
    return [
        {'params': rwkv_params, 'lr': base_lr, 'weight_decay': weight_decay, 'name': 'rwkv'},
        {'params': mamba_params, 'lr': base_lr * mamba_lr_mult, 'weight_decay': weight_decay, 'name': 'mamba'},
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay, 'name': 'other'},
    ]


# ============ Gradient Scaling ============
def register_gradient_scaling(model, mamba_grad_scale=1.0):
    """
    Register backward hooks to scale Mamba gradients during backward pass.
    
    This is different from LR multiplier:
    - LR mult: Changes step size in optimizer (grad * lr)
    - Grad scale: Changes gradient magnitude BEFORE optimizer sees it
    
    Use case: RWKV produces ~10x larger gradients than Mamba due to "easier" 
    signal paths. Gradient scaling directly compensates for this imbalance.
    
    Args:
        model: The model to register hooks on
        mamba_grad_scale: Scale factor for Mamba gradients (default 1.0 = no scaling)
        
    Returns:
        List of hook handles (for removal if needed)
    """
    if mamba_grad_scale == 1.0:
        return []  # No scaling needed
    
    handles = []
    scaled_count = 0
    
    def make_hook(scale):
        def hook(grad):
            return grad * scale
        return hook
    
    for name, param in model.named_parameters():
        if param.requires_grad and 'mamba' in name.lower():
            handle = param.register_hook(make_hook(mamba_grad_scale))
            handles.append(handle)
            scaled_count += 1
    
    print(f"  Gradient scaling: {scaled_count} Mamba params scaled by {mamba_grad_scale}x")
    return handles


# ============ LR Schedule ============
def get_lr_lambda(warmup_steps, max_steps, min_lr_ratio=0.1):
    """
    Cosine decay with warmup. Returns lambda for LambdaLR.
    Per V4_DESIGN.md: warmup 2000 steps, decay to 10% of peak.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        # Cosine decay to min_lr
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    return lr_lambda


# ============ Component Gradient Monitoring ============
# Per V3_CROSS_REFERENCE.md Section 9.4: Track RWKV vs Mamba gradient norms
def get_component_gradients(model):
    """
    Compute gradient norms for RWKV and Mamba components separately.
    Returns (rwkv_norm, mamba_norm, ratio).
    Ratio 0.3-3.0 = OK, <0.1 or >10 = FAIL per Cross-Ref 1.2
    """
    rwkv_grads = []
    mamba_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            if 'rwkv' in name.lower():
                rwkv_grads.append(grad_norm ** 2)
            elif 'mamba' in name.lower():
                mamba_grads.append(grad_norm ** 2)
    
    rwkv_norm = math.sqrt(sum(rwkv_grads)) if rwkv_grads else 0.0
    mamba_norm = math.sqrt(sum(mamba_grads)) if mamba_grads else 0.0
    
    # Avoid division by zero
    if mamba_norm < 1e-10:
        ratio = float('inf') if rwkv_norm > 1e-10 else 1.0
    else:
        ratio = rwkv_norm / mamba_norm
    
    return rwkv_norm, mamba_norm, ratio


def check_gradient_health(ratio, warn_threshold=3.0, fail_threshold=10.0):
    """
    Check if component gradient ratio is healthy.
    Returns: ('OK', None) | ('WARN', msg) | ('FAIL', msg)
    """
    if ratio < 1.0 / fail_threshold or ratio > fail_threshold:
        return ('FAIL', f'Gradient ratio {ratio:.2f} outside [0.1, 10] - one component dead!')
    elif ratio < 1.0 / warn_threshold or ratio > warn_threshold:
        return ('WARN', f'Gradient ratio {ratio:.2f} outside [0.33, 3.0] - imbalance')
    return ('OK', None)


# ============ Activation Monitoring ============
# Per V3_RESEARCH_NOTES.md Section 2.19: Monitor hidden state health

def compute_activation_stats(all_activations):
    """
    Compute statistics from layer activations for health monitoring.
    
    Args:
        all_activations: list of dicts, each with 'rwkv' and 'mamba' tensors
        
    Returns:
        dict with entropy, variance, norm stats per component
    """
    rwkv_stats = {'vars': [], 'norms': [], 'stds': []}
    mamba_stats = {'vars': [], 'norms': [], 'stds': []}
    
    for layer_acts in all_activations:
        # RWKV stats
        rwkv_out = layer_acts['rwkv'].detach()
        rwkv_stats['vars'].append(rwkv_out.var().item())
        rwkv_stats['norms'].append(rwkv_out.norm().item())
        rwkv_stats['stds'].append(rwkv_out.std().item())
        
        # Mamba stats
        mamba_out = layer_acts['mamba'].detach()
        mamba_stats['vars'].append(mamba_out.var().item())
        mamba_stats['norms'].append(mamba_out.norm().item())
        mamba_stats['stds'].append(mamba_out.std().item())
    
    return {
        'rwkv': {
            'mean_var': sum(rwkv_stats['vars']) / len(rwkv_stats['vars']),
            'mean_norm': sum(rwkv_stats['norms']) / len(rwkv_stats['norms']),
            'mean_std': sum(rwkv_stats['stds']) / len(rwkv_stats['stds']),
        },
        'mamba': {
            'mean_var': sum(mamba_stats['vars']) / len(mamba_stats['vars']),
            'mean_norm': sum(mamba_stats['norms']) / len(mamba_stats['norms']),
            'mean_std': sum(mamba_stats['stds']) / len(mamba_stats['stds']),
        }
    }


def compute_hidden_entropy(hidden_states):
    """
    Compute Shannon entropy of hidden states to detect collapse/chaos.
    Per V3_RESEARCH_NOTES.md Section 2.19.
    
    Args:
        hidden_states: tensor of shape [batch, seq, hidden]
        
    Returns:
        float: mean entropy value
        
    Interpretation:
        < 1.0 = State Collapse (frozen)
        2.0 - 5.0 = Healthy
        > 7.0 = State Chaos
    """
    # Normalize to pseudo-probability distribution
    probs = F.softmax(hidden_states.detach(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.mean().item()


def check_activation_health(stats, entropy):
    """
    Check activation statistics for warning signs.
    
    Returns: list of (level, message) tuples
    """
    warnings = []
    
    # Check for collapsed component (near-zero std)
    if stats['rwkv']['mean_std'] < 0.01:
        warnings.append(('FAIL', f"RWKV activations collapsed! std={stats['rwkv']['mean_std']:.4f}"))
    if stats['mamba']['mean_std'] < 0.01:
        warnings.append(('FAIL', f"Mamba activations collapsed! std={stats['mamba']['mean_std']:.4f}"))
    
    # Check variance ratio between components
    var_ratio = stats['rwkv']['mean_var'] / (stats['mamba']['mean_var'] + 1e-10)
    if var_ratio < 0.1 or var_ratio > 10:
        warnings.append(('WARN', f"Activation variance ratio {var_ratio:.2f} - component imbalance"))
    
    # Check entropy
    if entropy < 1.0:
        warnings.append(('FAIL', f"State COLLAPSE detected! entropy={entropy:.2f}"))
    elif entropy > 7.0:
        warnings.append(('WARN', f"State CHAOS detected! entropy={entropy:.2f}"))
    
    return warnings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train GroundThink hybrid model')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config file (e.g., configs/train_8m_50k.yaml)')
    
    # Common overrides
    parser.add_argument('--model', type=str, default=None, 
                        help=f'Model name: {list(list_models().keys())}')
    parser.add_argument('--max-steps', type=int, default=None, dest='max_steps',
                        help='Training steps')
    parser.add_argument('--batch-size', type=int, default=None, dest='batch_size',
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--mamba-grad-scale', type=float, default=None, dest='mamba_grad_scale',
                        help='Gradient scale for Mamba params (default 1.0, try 10.0 for balance)')
    parser.add_argument('--rwkv-dropout', type=float, default=None, dest='rwkv_dropout_start',
                        help='RWKV dropout start prob (e.g., 0.4 = 40%% chance of suppressing RWKV)')
    parser.add_argument('--data', type=str, default='data/shakespeare.txt',
                        help='Training data file (default: data/shakespeare.txt)')
    parser.add_argument('--tokenizer', type=str, default='char', choices=['char', 'bpe'],
                        help='Tokenizer type: char (default) or bpe')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Checkpoint to resume from')
    parser.add_argument('--log-states', action='store_true', dest='log_states',
                        help='Enable internal state monitoring (S0-S4 diagnostics)')
    
    args = parser.parse_args()
    
    # Build CLI overrides dict (only non-None values)
    cli_overrides = {
        'model': args.model,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'mamba_grad_scale': args.mamba_grad_scale,
        'rwkv_dropout_start': args.rwkv_dropout_start,
        'rwkv_dropout_end': 0.1 if args.rwkv_dropout_start else None,  # Anneal to 10% if enabled
        'log_states': args.log_states,
    }
    
    # Load config with overrides
    CONFIG = load_config(args.config, cli_overrides)
    
    print(f"=== GroundThink Training ===")
    print(f"Model: {CONFIG['model']}")
    print(f"Steps: {CONFIG['max_steps']}, Batch: {CONFIG['batch_size']}, LR: {CONFIG['lr']}")
    if args.config:
        print(f"Config: {args.config}")
    
    # Load data
    print(f"\nLoading dataset from {args.data}...")
    
    # Determine scale for tokenizer selection
    if args.tokenizer == 'bpe':
        tok_scale = 'LARGE'  # Forces BPE tokenizer
    else:
        tok_scale = CONFIG['model']  # Uses char-level for small models
    
    dataset, tokenizer = load_stateful_dataset(
        args.data,
        batch_size=CONFIG['batch_size'],
        seq_len=CONFIG['seq_len'],
        scale=tok_scale,
    )
    
    # Create model from registry
    print(f"\nCreating model '{CONFIG['model']}' (vocab={tokenizer.vocab_size})...")
    model = get_model(CONFIG['model'], vocab_size=tokenizer.vocab_size).to(device)
    print(f"Model params: {model.get_num_params():,}")
    
    # Setup optimizer with differential LR
    param_groups = get_parameter_groups(
        model, 
        base_lr=CONFIG['lr'],
        mamba_lr_mult=CONFIG['mamba_lr_mult'],
        weight_decay=CONFIG['weight_decay'],
    )
    optimizer = torch.optim.AdamW(param_groups, betas=CONFIG['betas'])
    
    # Print param group info
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: {n_params:,} params, lr={pg['lr']:.1e}")
    
    # Compute warmup steps from ratio (V3 guideline: 5-10% of max_steps)
    warmup_steps = int(CONFIG['max_steps'] * CONFIG['warmup_ratio'])
    
    # Setup LR scheduler
    lr_lambda = get_lr_lambda(
        warmup_steps=warmup_steps,
        max_steps=CONFIG['max_steps'],
        min_lr_ratio=CONFIG['min_lr'] / CONFIG['lr'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("\nOptimizer and scheduler ready!")
    print(f"Warmup: {warmup_steps} steps ({CONFIG['warmup_ratio']*100:.0f}% of {CONFIG['max_steps']})")
    print(f"LR: {CONFIG['lr']} -> {CONFIG['min_lr']} (cosine)")
    
    # Register gradient scaling for Mamba (if enabled)
    grad_scale_handles = register_gradient_scaling(model, CONFIG['mamba_grad_scale'])
    
    # Resume from checkpoint if specified
    start_step = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"\n=== Resuming from {args.resume} ===")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt['step']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")
    
    # Quick forward+backward test (skip if resuming)
    if args.resume is None:
        print("\n=== Testing forward+backward ===")
        model.train()
        x, y, _ = dataset[0]
        x, y = x.to(device), y.to(device)
    
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        print(f"Forward OK, loss = {loss.item():.4f}")
    
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        print(f"Backward OK, grad_norm = {grad_norm:.4f}")
    
        optimizer.step()
        scheduler.step()
        print("Optimizer step OK")
    
        # VRAM usage
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"VRAM used: {vram_mb:.0f} MB")
    
    # ============ Validation Function ============
    @torch.no_grad()
    def evaluate_val_loss(model, dataset, n_batches=10):
        """Compute validation loss on n_batches from validation set."""
        model.eval()
        dataset.reset_val()  # Reset to beginning each time
        total_loss = 0.0
        count = 0
        for i in range(n_batches):
            batch = dataset.get_val_batch()
            if batch is None:
                break  # Exhausted val set
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1
        model.train()
        return total_loss / max(1, count)

    # ============ Activation Monitoring Function ============
    @torch.no_grad()
    def get_activation_diagnostics(model, x):
        """Run forward pass with activation capture for diagnostics."""
        model.eval()
        logits, all_activations = model(x, return_activations=True)
        
        # Get hidden states from last layer (before final norm)
        # Use the combined output after fusion for entropy
        last_layer_rwkv = all_activations[-1]['rwkv']
        last_layer_mamba = all_activations[-1]['mamba']
        hidden_states = last_layer_rwkv + last_layer_mamba  # Combined signal
        
        stats = compute_activation_stats(all_activations)
        entropy = compute_hidden_entropy(hidden_states)
        
        model.train()
        return stats, entropy

    # ============ State Monitoring Function (Task 50, enhanced Task 57) ============
    # Import tools for enhanced diagnostics
    try:
        from tools.thresholds import check_status, THRESHOLDS
        from tools.state_metrics import StateMetrics
        from tools.gradient_coupling import GradientCoupling
        state_tracker = StateMetrics()
        grad_analyzer = None  # Initialized with model later
        TOOLS_AVAILABLE = True
    except ImportError:
        TOOLS_AVAILABLE = False
        state_tracker = None
        grad_analyzer = None
    
    @torch.no_grad()
    def get_state_diagnostics(model, x):
        """Capture internal states for S0-S4 style monitoring.
        
        Enhanced (Task 57) to integrate:
            - StateMetrics (Task 53) for tracking
            - Threshold checks (Task 56) for status
        
        Returns dict with:
            - rwkv_state_norm: L2 norm of RWKV recurrent state
            - mamba_state_norm: L2 norm of Mamba output proxy
            - state_ratio: rwkv_norm / mamba_norm
            - gate: fusion gate value (model-dependent)
            - status: PASS/WARN/FAIL based on thresholds
        """
        model.eval()
        try:
            logits, states = model(x, return_states=True)
        except TypeError:
            # Model doesn't support return_states
            model.train()
            return None
        
        rwkv_state = states.get('rwkv_state')
        mamba_state = states.get('mamba_state')
        
        rwkv_norm = rwkv_state.norm().item() if rwkv_state is not None else 0.0
        mamba_norm = mamba_state.norm().item() if mamba_state is not None else 0.0
        
        # Avoid division by zero
        state_ratio = rwkv_norm / max(mamba_norm, 1e-8)
        
        # Determine status using unified thresholds (Task 56)
        if TOOLS_AVAILABLE:
            status = check_status('s4_state_ratio', state_ratio)
        else:
            # Fallback if tools not available
            if state_ratio > 10.0:
                status = 'FAIL'
            elif state_ratio > 3.0:
                status = 'WARN'
            else:
                status = 'PASS'
        
        diagnostics = {
            'rwkv_state_norm': rwkv_norm,
            'mamba_state_norm': mamba_norm,
            'state_ratio': state_ratio,
            'gate': states.get('gate', 0.5),
            'status': status,
        }
        
        model.train()
        return diagnostics

    # Training loop with monitoring
    resume_info = f" (resuming from {start_step})" if start_step > 0 else ""
    print(f"\n=== Training 5000 steps (HY Fusion){resume_info} ===", flush=True)
    if CONFIG.get('use_amp', False):
        print("Mode: Mixed Precision (AMP)")
    
    # RWKV Dropout setup (annealing schedule)
    rwkv_drop_start = CONFIG.get('rwkv_dropout_start', 0.0)
    rwkv_drop_end = CONFIG.get('rwkv_dropout_end', 0.0)
    if rwkv_drop_start > 0:
        print(f"RWKV Dropout: {rwkv_drop_start:.0%} -> {rwkv_drop_end:.0%} (linear anneal)")
    
    import sys
    import time
    start_time = time.time()
    tokens_processed = 0
    
    model.train()
    step = start_step
    total_loss = 0.0
    # best_val_loss already set from checkpoint or inf
    val_no_improve = 0
    log_interval_start = time.time()
    log_interval_tokens = 0
    
    max_steps = CONFIG['max_steps']
    log_every = CONFIG['log_every']
    save_every = CONFIG['save_every']
    
    # AMP setup
    use_amp = CONFIG.get('use_amp', False)
    scaler = GradScaler('cuda') if use_amp else None
    
    for epoch in range(1000):  # Enough epochs
        for idx in range(len(dataset)):
            if step >= max_steps:
                break
            
            x, y, _ = dataset[idx]
            x, y = x.to(device), y.to(device)
            batch_tokens = x.numel()
            
            # Compute current RWKV dropout probability (linear anneal)
            progress = step / max(1, max_steps - 1)
            rwkv_drop_prob = rwkv_drop_start + (rwkv_drop_end - rwkv_drop_start) * progress
            
            optimizer.zero_grad()
            
            # Forward with optional AMP and RWKV dropout
            with autocast('cuda', enabled=use_amp):
                logits = model(x, rwkv_drop_prob=rwkv_drop_prob)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward with optional scaler
            if use_amp:
                scaler.scale(loss).backward()
                # Component gradient check (before clipping)
                scaler.unscale_(optimizer)
                rwkv_norm, mamba_norm, ratio = get_component_gradients(model)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                rwkv_norm, mamba_norm, ratio = get_component_gradients(model)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            tokens_processed += batch_tokens
            log_interval_tokens += batch_tokens
            step += 1
            
            # Quick progress indicator every 10 steps
            if step % 10 == 0:
                print(f".", end="", flush=True)
            
            # Logging every log_every steps
            if step % log_every == 0:
                avg_loss = total_loss / log_every
                lr = scheduler.get_last_lr()[0]
                val_loss = evaluate_val_loss(model, dataset, n_batches=5)
                
                # Token/s calculation
                elapsed = time.time() - log_interval_start
                tok_per_sec = log_interval_tokens / elapsed if elapsed > 0 else 0
                
                # Get activation diagnostics
                act_stats, entropy = get_activation_diagnostics(model, x)
                
                # Check gradient health
                status, msg = check_gradient_health(ratio, 
                    CONFIG['grad_ratio_warn'], CONFIG['grad_ratio_fail'])
                status_str = f" [{status}]" if status != 'OK' else ""
                
                # Check activation health
                act_warnings = check_activation_health(act_stats, entropy)
                
                # Perplexity = exp(loss)
                train_ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
                val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
                
                print(f"Step {step:4d}/{max_steps} | Loss: {avg_loss:.4f}/{val_loss:.4f} | "
                      f"PPL: {train_ppl:.2f}/{val_ppl:.2f} | R/M: {ratio:.2f}{status_str} | "
                      f"H: {entropy:.2f} | tok/s: {tok_per_sec:.0f} | LR: {lr:.2e}", flush=True)
                
                # Print gradient warning
                if msg:
                    print(f"  ⚠️ GRAD: {msg}", flush=True)
                
                # Print activation warnings
                for level, warn_msg in act_warnings:
                    icon = "🔴" if level == 'FAIL' else "⚠️"
                    print(f"  {icon} ACT: {warn_msg}", flush=True)
                
                # Print activation details every 500 steps
                if step % 500 == 0:
                    print(f"    RWKV: var={act_stats['rwkv']['mean_var']:.4f} std={act_stats['rwkv']['mean_std']:.4f}", flush=True)
                    print(f"    Mamba: var={act_stats['mamba']['mean_var']:.4f} std={act_stats['mamba']['mean_std']:.4f}", flush=True)
                
                # State monitoring (enabled with --log-states) - Enhanced Task 57
                if CONFIG.get('log_states', False):
                    state_diag = get_state_diagnostics(model, x)
                    if state_diag is not None:
                        status_icon = {'PASS': '✓', 'WARN': '⚠️', 'FAIL': '🔴'}.get(state_diag['status'], '?')
                        print(f"    STATE: rwkv={state_diag['rwkv_state_norm']:.1f} mamba={state_diag['mamba_state_norm']:.1f} "
                              f"ratio={state_diag['state_ratio']:.1f}x gate={state_diag['gate']:.3f} {status_icon}{state_diag['status']}", flush=True)
                        
                        # Track with StateMetrics if available
                        if TOOLS_AVAILABLE and state_tracker is not None:
                            state_tracker.log(step, {
                                'rwkv_state': torch.tensor([state_diag['rwkv_state_norm']]),
                                'mamba_state': torch.tensor([state_diag['mamba_state_norm']]),
                            })
                
                # Track best val
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    val_no_improve = 0
                else:
                    val_no_improve += 1
                
                total_loss = 0.0
                log_interval_start = time.time()
                log_interval_tokens = 0
            
            # Checkpointing
            if step % save_every == 0:
                model_name = CONFIG['model']
                ckpt_path = f"checkpoints/ckpt_{model_name}_step{step}.pt"
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': CONFIG,
                }, ckpt_path)
                print(f"  💾 Saved checkpoint: {ckpt_path}", flush=True)
        
        if step >= max_steps:
            break
    
    # Final summary
    elapsed = time.time() - start_time
    total_tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
    model_name = CONFIG['model']
    print(f"\n{'='*60}", flush=True)
    print(f"TRAINING COMPLETE: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Steps: {step}", flush=True)
    print(f"Time: {elapsed:.1f}s ({step/elapsed:.2f} steps/s)", flush=True)
    print(f"Tokens: {tokens_processed:,} ({total_tok_per_sec:.0f} tok/s avg)", flush=True)
    print(f"Best val loss: {best_val_loss:.4f}", flush=True)
    print(f"Val no-improve count: {val_no_improve}", flush=True)
    
    # Save final checkpoint
    final_path = f"checkpoints/ckpt_{model_name}_final.pt"
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': CONFIG,
    }, final_path)
    print(f"💾 Final checkpoint: {final_path}", flush=True)

