"""
⛔ DEPRECATED - SEE V3_DEPRECATED.md

Training loop for wrong architecture. Data/validation logic may be salvageable.

---

GroundThink V3 Training Script (DEPRECATED)

Implements stateful training per V3_RESEARCH_NOTES.md:
- State-handoff training loop (Section 2.11)
- Optimizer parameter groups (Section 2.15)
- State entropy monitoring (Section 2.19)
- Curriculum learning support (Section 2.18, 2.23)
- Entropy regularized loss (Section 2.27)

Usage:
  python train_v030.py --config 8M_v3
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from pathlib import Path
import math
import json

from layers_v030 import GroundThinkV3, get_layer_config, VERSION
from data_v030 import load_stateful_dataset, CharTokenizer


# ============================================================================
# CURRICULUM LEARNING (Section 2.18, 2.23)
# ============================================================================

def get_curriculum_seq_len(step: int, total_steps: int) -> int:
    """
    Get sequence length for current step per Section 2.18 Grow-P2 Schedule.
    
    Simple 4-phase curriculum:
    - Phase 1 (0-5%):   128 tokens - Build basic pattern recognition
    - Phase 2 (5-25%):  256 tokens - Learn local dependencies
    - Phase 3 (25-75%): 512 tokens - Develop state compression
    - Phase 4 (75-100%): 1024 tokens - Master long-range identity
    """
    progress = step / max(total_steps, 1)
    
    if progress < 0.05:
        return 128
    elif progress < 0.25:
        return 256
    elif progress < 0.75:
        return 512
    else:
        return 1024


def curriculum_transition(
    optimizer: torch.optim.Optimizer,
    old_seq_len: int,
    new_seq_len: int,
    old_batch_size: int,
    old_lr: float,
) -> tuple[int, float]:
    """
    Calculate new hyperparameters when transitioning curriculum phases.
    Per Section 2.23: Double batch size, halve LR when seq_len increases.
    
    Returns:
        new_batch_size: Adjusted batch size
        new_lr: Adjusted learning rate
    """
    if new_seq_len == old_seq_len:
        return old_batch_size, old_lr
    
    scale_factor = new_seq_len / old_seq_len
    
    new_batch_size = int(old_batch_size * scale_factor)
    new_lr = old_lr / scale_factor
    
    # Update optimizer LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / scale_factor
    
    print(f"📚 Curriculum transition: {old_seq_len} → {new_seq_len}")
    print(f"   Batch size: {old_batch_size} → {new_batch_size}")
    print(f"   Learning rate: {old_lr:.2e} → {new_lr:.2e}")
    
    return new_batch_size, new_lr


# ============================================================================
# ENTROPY REGULARIZED LOSS (Section 2.27)
# ============================================================================

def entropy_regularized_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    entropy_weight: float = 0.01,
    ignore_index: int = 0,
) -> tuple[torch.Tensor, float, float]:
    """
    Standard CE loss + entropy bonus to prevent mode collapse.
    Per Section 2.27.
    
    Args:
        logits: Model output [batch, seq, vocab]
        targets: Target tokens [batch, seq]
        entropy_weight: Weight for entropy regularization (default 0.01)
        ignore_index: Token ID to ignore in loss (default 0 = PAD)
        
    Returns:
        total_loss: Combined loss for backprop
        ce_loss: Cross-entropy loss value (for logging)
        entropy: Output entropy value (for logging)
    """
    # Standard cross-entropy
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        targets.view(-1),
        ignore_index=ignore_index,
    )
    
    # Entropy of output distribution (higher = more diverse)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
    
    # We SUBTRACT entropy to encourage diversity (minimize negative entropy)
    total_loss = ce_loss - entropy_weight * entropy
    
    return total_loss, ce_loss.item(), entropy.item()


def check_mode_collapse(
    model: nn.Module,
    logits: torch.Tensor,
    step: int,
) -> bool:
    """
    Detect mode collapse per Section 2.27 diagnostics.
    
    Returns True if collapse detected.
    """
    # Check embedding norm
    embed_norm = model.embed.weight.norm().item()
    if embed_norm > 50.0:
        print(f"⚠️ Step {step}: Embedding norm={embed_norm:.1f} (EXPLOSION)")
        return True
    
    # Check top-1 probability domination
    probs = F.softmax(logits, dim=-1)
    top1_prob = probs.max(dim=-1).values.mean().item()
    if top1_prob > 0.95:
        print(f"⚠️ Step {step}: Top-1 prob={top1_prob:.3f} (MODE COLLAPSE)")
        return True
    
    return False


# ============================================================================
# MONITORING FUNCTIONS (Section 2.19, 2.32)
# ============================================================================

def compute_state_entropy(states: list[torch.Tensor]) -> float:
    """
    Compute entropy of hidden states to detect collapse or chaos.
    
    From Section 2.19:
    - Low entropy (< 1.0) = State Collapse
    - Healthy entropy (2.0 - 5.0) = Good utilization
    - High entropy (> 7.0) = State Chaos
    """
    if not states or states[0] is None:
        return 0.0
    
    entropies = []
    for state in states:
        if state is None:
            continue
        # Flatten and compute pseudo-probability distribution
        flat = state.detach().view(-1)
        probs = F.softmax(flat, dim=0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        entropies.append(entropy.item())
    
    return sum(entropies) / len(entropies) if entropies else 0.0


def compute_state_norms(states: list[torch.Tensor]) -> list[float]:
    """
    Compute RMS (root mean square) of each layer's state.
    
    Returns RMS = norm / sqrt(numel), which indicates typical magnitude.
    Healthy range: 0.5 - 2.0
    If > 10: State explosion
    If < 0.01: State vanishing
    """
    rms_values = []
    for state in states:
        if state is not None:
            # RMS = Frobenius norm / sqrt(number of elements)
            rms = state.norm().item() / (state.numel() ** 0.5)
            rms_values.append(rms)
    return rms_values


def check_phase_shift(model: nn.Module, step: int) -> float:
    """
    Monitor for Phase Shift moment per Section 2.32.
    Returns ratio of delta gradient norm to other gradients.
    """
    delta_grads = []
    other_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'time_decay' in name.lower() or 'gate' in name.lower():
                delta_grads.append(grad_norm)
            else:
                other_grads.append(grad_norm)
    
    avg_delta = sum(delta_grads) / len(delta_grads) if delta_grads else 0
    avg_other = sum(other_grads) / len(other_grads) if other_grads else 1
    
    ratio = avg_delta / (avg_other + 1e-9)
    
    if ratio > 5.0:
        print(f"🎯 Step {step}: PHASE SHIFT! Delta grad ratio = {ratio:.2f}")
    
    return ratio


def compute_validation_loss(
    model: nn.Module,
    dataset,
    states: list[torch.Tensor] | None,
    device: str,
    max_batches: int = 10,
) -> float:
    """
    Compute validation loss over validation batches.
    
    Args:
        model: The model to evaluate
        dataset: StatefulDataset with get_val_batch() method
        states: Current model states (not modified)
        device: Device to run on
        max_batches: Max validation batches to evaluate (for speed)
        
    Returns:
        Average validation cross-entropy loss
    """
    model.eval()
    dataset.reset_val()
    
    total_loss = 0.0
    num_batches = 0
    
    # Use fresh states for validation (don't corrupt training states)
    val_states = None
    
    with torch.no_grad():
        while num_batches < max_batches:
            batch = dataset.get_val_batch()
            if batch is None:
                break
            
            x, y, is_new_doc = batch
            x, y = x.to(device), y.to(device)
            
            # Reset states for new documents in validation
            if val_states is not None:
                for b in range(x.shape[0]):
                    if is_new_doc[b]:
                        for layer_state in val_states:
                            if layer_state is not None:
                                layer_state[b].zero_()
            
            logits, val_states = model(x, val_states)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=0,
            )
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    
    if num_batches == 0:
        return 0.0
    
    return total_loss / num_batches


# ============================================================================
# TRAINING LOOP (Section 2.11)
# ============================================================================

def stateful_train_step(
    model: nn.Module,
    batch: tuple,
    states: list[torch.Tensor] | None,
    optimizer: torch.optim.Optimizer,
    use_entropy_reg: bool = True,
    entropy_weight: float = 0.01,
    ignore_index: int = 0,
) -> tuple[float, list[torch.Tensor], float, float, float]:
    """
    Single stateful training step with entropy regularization.
    
    Returns:
        loss: Total loss value (CE - entropy*weight)
        new_states: Updated states (detached for next step)
        grad_norm: Gradient norm after clipping
        ce_loss: Cross-entropy loss (for logging)
        entropy: Output entropy (for logging)
    """
    x, y, is_new_doc = batch
    device = x.device
    B = x.shape[0]
    
    # Reset states for new documents
    if states is not None:
        for b in range(B):
            if is_new_doc[b]:
                for layer_state in states:
                    if layer_state is not None:
                        layer_state[b].zero_()
    
    optimizer.zero_grad()
    
    # Forward pass
    logits, new_states = model(x, states)
    
    # Compute loss with optional entropy regularization
    if use_entropy_reg:
        loss, ce_loss, entropy = entropy_regularized_loss(
            logits, y, entropy_weight, ignore_index
        )
    else:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.view(-1),
            ignore_index=ignore_index,
        )
        ce_loss = loss.item()
        entropy = 0.0
    
    # Backward pass
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Detach states for next step (TBPTT)
    detached_states = []
    for state in new_states:
        if state is not None:
            detached_states.append(state.detach())
        else:
            detached_states.append(None)
    
    return loss.item(), detached_states, grad_norm.item(), ce_loss, entropy


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='8M_v3', help='Config name')
    parser.add_argument('--steps', type=int, default=None, help='Override num_steps')
    parser.add_argument('--data', type=str, default='shakespeare.txt', 
                        help='Data file (default: shakespeare.txt for G3 validation)')
    parser.add_argument('--no-entropy-reg', action='store_true',
                        help='Disable entropy regularization')
    parser.add_argument('--entropy-weight', type=float, default=0.01,
                        help='Entropy regularization weight (default: 0.01)')
    args = parser.parse_args()
    
    # Load config
    try:
        config_module = __import__(f'configs.model_{args.config}', fromlist=[''])
        model_cfg = config_module.MODEL_CONFIG
        train_cfg = config_module.TRAIN_CONFIG.copy()
        save_name = config_module.SAVE_NAME
    except ModuleNotFoundError:
        print(f"Config '{args.config}' not found, using defaults")
        model_cfg = dict(n_layers=12, dim=256, n_heads=8, head_dim=32, state_dim=16)
        train_cfg = dict(seq_len=256, batch_size=8, num_steps=5000, lr_base=6e-4, warmup=500)
        save_name = f"groundthink_v3_{args.config}"
    
    if args.steps:
        train_cfg['num_steps'] = args.steps
    
    print(f"GroundThink V{VERSION}")
    print(f"Config: {json.dumps(model_cfg, indent=2)}")
    print(f"Training: {json.dumps(train_cfg, indent=2)}")
    print(f"Entropy Reg: {'OFF' if args.no_entropy_reg else f'ON (weight={args.entropy_weight})'}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return
    
    dataset, tokenizer = load_stateful_dataset(
        data_path, 
        train_cfg['batch_size'], 
        train_cfg['seq_len']
    )
    
    # Create model
    attn_positions = model_cfg.get('attn_positions', [model_cfg['n_layers'] // 2])
    
    model = GroundThinkV3(
        vocab_size=tokenizer.vocab_size,
        n_layers=model_cfg['n_layers'],
        dim=model_cfg['dim'],
        n_heads=model_cfg['n_heads'],
        head_dim=model_cfg['head_dim'],
        state_dim=model_cfg.get('state_dim', 16),
        attn_positions=attn_positions,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Optimizer with parameter groups
    param_groups = model.get_param_groups(train_cfg['lr_base'], weight_decay=0.1)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    
    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: {n:,} params, lr={pg['lr']:.1e}")
    
    # LR scheduler
    def lr_lambda(step):
        if step < train_cfg['warmup']:
            return step / train_cfg['warmup']
        progress = (step - train_cfg['warmup']) / (train_cfg['num_steps'] - train_cfg['warmup'])
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training settings
    use_entropy_reg = not args.no_entropy_reg
    entropy_weight = args.entropy_weight
    
    # Training loop
    print(f"\nTraining for {train_cfg['num_steps']} steps...")
    print(f"Gate G3 Targets: loss decreasing, grad norm 0.5-1.5")
    print("-" * 60)
    model.train()
    
    states = None
    step = 0
    total_loss = 0
    total_ce = 0
    total_ent = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    start_time = time.time()
    collapse_detected = False
    
    # Track for G3 validation
    first_loss = None
    grad_norms = []
    
    # Track validation losses for checkpoint
    val_loss_history = []
    
    while step < train_cfg['num_steps']:
        for idx in range(len(dataset)):
            if step >= train_cfg['num_steps']:
                break
            
            x, y, is_new_doc = dataset[idx]
            x, y = x.to(device), y.to(device)
            is_new_doc = is_new_doc.to(device)
            
            loss, states, grad_norm, ce_loss, entropy = stateful_train_step(
                model, (x, y, is_new_doc), states, optimizer,
                use_entropy_reg=use_entropy_reg,
                entropy_weight=entropy_weight,
            )
            
            # Track first loss for G3 validation
            if first_loss is None:
                first_loss = ce_loss
            
            # Track grad norms for G3 validation
            grad_norms.append(grad_norm)
            
            scheduler.step()
            total_loss += loss
            total_ce += ce_loss
            total_ent += entropy
            step += 1
            
            # Track best loss every step
            if ce_loss < best_loss:
                best_loss = ce_loss
            
            # Check for mode collapse every 100 steps
            if step % 100 == 0:
                with torch.no_grad():
                    logits, _ = model(x, states)
                    if check_mode_collapse(model, logits, step):
                        collapse_detected = True
            
            # Logging
            if step % 100 == 0:
                avg_loss = total_loss / 100
                avg_ce = total_ce / 100
                avg_ent = total_ent / 100
                elapsed = time.time() - start_time
                tok_s = step * train_cfg['batch_size'] * train_cfg['seq_len'] / elapsed
                lr = scheduler.get_last_lr()[0]
                
                # Compute validation loss every 100 steps
                val_loss = compute_validation_loss(model, dataset, states, device)
                val_loss_history.append({'step': step, 'val_loss': val_loss, 'train_loss': avg_ce})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # State diagnostics every 500 steps
                extra = ""
                if step % 500 == 0:
                    state_entropy = compute_state_entropy(states)
                    norms = compute_state_norms(states)
                    extra = f"\n       State ent={state_entropy:.1f} norms={[f'{n:.1f}' for n in norms[:3]]}"
                
                ent_str = f" Ent:{avg_ent:.2f}" if use_entropy_reg else ""
                print(f"Step {step:5d} | Train Loss: {avg_ce:.4f} | Val Loss: {val_loss:.4f}{ent_str} | "
                      f"LR:{lr:.2e} | Grad:{grad_norm:.2f} | {tok_s:.0f} tok/s{extra}")
                
                total_loss = 0
                total_ce = 0
                total_ent = 0
        
        # Reset for next epoch
        states = None
    
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Done! {elapsed:.1f}s")
    print(f"Best Train Loss: {best_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")
    
    # Gate G3 Validation Report
    print("\n" + "=" * 60)
    print("GATE G3 VALIDATION REPORT")
    print("=" * 60)
    
    loss_decreased = best_loss < first_loss
    avg_grad = sum(grad_norms) / len(grad_norms)
    grad_in_range = 0.5 <= avg_grad <= 1.5
    
    print(f"1. Loss Decreasing: {first_loss:.4f} → {best_loss:.4f}")
    print(f"   {'✅ PASS' if loss_decreased else '❌ FAIL'}: Loss {'decreased' if loss_decreased else 'did NOT decrease'}")
    
    print(f"2. Grad Norm: avg={avg_grad:.3f} (target: 0.5-1.5)")
    print(f"   {'✅ PASS' if grad_in_range else '⚠️ WARN'}: Grad norm {'in' if grad_in_range else 'outside'} target range")
    
    print(f"3. Mode Collapse: {'❌ DETECTED' if collapse_detected else '✅ None detected'}")
    
    print(f"4. Validation Loss: Best={best_val_loss:.4f}")
    
    g3_passed = loss_decreased and not collapse_detected
    print(f"\nGATE G3: {'✅ PASSED' if g3_passed else '❌ FAILED'}")
    print("=" * 60)
    
    # Save checkpoint
    save_path = f"{save_name}_{train_cfg['num_steps']//1000}k.pt"
    torch.save({
        'model': model.state_dict(),
        'vocab': tokenizer.char_to_id,
        'model_config': model_cfg,
        'train_config': train_cfg,
        'version': VERSION,
        'best_loss': best_loss,
        'best_val_loss': best_val_loss,
        'val_loss_history': val_loss_history,
    }, save_path)
    print(f"✅ Saved to {save_path}")


if __name__ == '__main__':
    main()
