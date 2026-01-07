"""
REAL TRAINING LOOP FOR HYBRID SSMs
This is what actually works, not what papers say works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
from tqdm import tqdm
import wandb  # optional, but useful

# ========== SETUP LOGGING (CRITICAL) ==========

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_gpu_memory():
    """Log GPU memory usage - essential for debugging"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: Allocated={memory_allocated:.2f}GB, Cached={memory_cached:.2f}GB")

# ========== REAL DATASET CLASS ==========

class TextDataset(Dataset):
    """
    Actual dataset that handles long contexts properly.
    Not the toy version from tutorials.
    """
    
    def __init__(self, filepaths, seq_len=2048, stride=512, tokenizer=None):
        self.seq_len = seq_len
        self.stride = stride
        
        # Load and tokenize all files
        self.tokens = []
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Simple tokenizer (replace with BPE in production)
            if tokenizer is None:
                # Character-level tokenizer for testing
                vocab = sorted(set(text))
                self.vocab_size = len(vocab)
                self.stoi = {ch: i for i, ch in enumerate(vocab)}
                self.itos = {i: ch for i, ch in enumerate(vocab)}
                tokens = [self.stoi[ch] for ch in text]
            else:
                tokens = tokenizer.encode(text)
            
            self.tokens.extend(tokens)
        
        self.tokens = np.array(self.tokens, dtype=np.int64)
        logger.info(f"Loaded dataset with {len(self.tokens)} tokens")
        
    def __len__(self):
        return (len(self.tokens) - self.seq_len) // self.stride
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len + 1  # +1 for target
        
        # Get sequence
        tokens = self.tokens[start:end]
        
        # Return input and target
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'target_ids': torch.tensor(tokens[1:], dtype=torch.long)
        }

# ========== GRADIENT DEBUGGING ==========

class GradientDebugger:
    """
    Tracks gradient issues in real-time.
    SSMs are notoriously gradient-unstable.
    """
    
    def __init__(self, model, log_interval=100):
        self.model = model
        self.log_interval = log_interval
        self.step_count = 0
        
        # Track gradient statistics
        self.grad_stats = {
            'mean': [],
            'std': [],
            'max': [],
            'min': [],
            'nan_count': 0,
            'inf_count': 0
        }
        
        # Hook into gradients
        self.register_hooks()
    
    def register_hooks(self):
        """Hook into all parameters to monitor gradients"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.grad_hook(grad, name))
    
    def grad_hook(self, grad, name):
        """Monitor gradient statistics"""
        self.step_count += 1
        
        if grad is None:
            return grad
        
        # Check for NaN/inf
        nan_mask = torch.isnan(grad)
        inf_mask = torch.isinf(grad)
        
        if nan_mask.any():
            self.grad_stats['nan_count'] += 1
            logger.warning(f"NaN gradient in {name} at step {self.step_count}")
            grad = torch.where(nan_mask, torch.zeros_like(grad), grad)
        
        if inf_mask.any():
            self.grad_stats['inf_count'] += 1
            logger.warning(f"Inf gradient in {name} at step {self.step_count}")
            grad = torch.where(inf_mask, torch.zeros_like(grad), grad)
        
        # Log statistics periodically
        if self.step_count % self.log_interval == 0:
            self.grad_stats['mean'].append(grad.mean().item())
            self.grad_stats['std'].append(grad.std().item())
            self.grad_stats['max'].append(grad.max().item())
            self.grad_stats['min'].append(grad.min().item())
            
            if self.step_count % (self.log_interval * 10) == 0:
                logger.info(f"Gradient stats (last {self.log_interval} steps):")
                logger.info(f"  Mean: {np.mean(self.grad_stats['mean'][-10:]):.6f}")
                logger.info(f"  Std: {np.mean(self.grad_stats['std'][-10:]):.6f}")
                logger.info(f"  NaN count: {self.grad_stats['nan_count']}")
                logger.info(f"  Inf count: {self.grad_stats['inf_count']}")
        
        return grad

# ========== STATE MONITOR ==========

class StateMonitor:
    """
    Monitors SSM and RWKV states for instability.
    """
    
    def __init__(self, model):
        self.model = model
        self.state_history = []
        self.gate_history = []
        
    def record_states(self, states):
        """Record state statistics for debugging"""
        stats = {}
        
        for name, state in states.items():
            if state is not None:
                # Compute various statistics
                stats[f"{name}_norm"] = torch.norm(state, dim=-1).mean().item()
                stats[f"{name}_mean"] = state.mean().item()
                stats[f"{name}_std"] = state.std().item()
                stats[f"{name}_max"] = state.max().item()
                stats[f"{name}_min"] = state.min().item()
                
                # Check for instability
                if stats[f"{name}_norm"] > 100:
                    logger.warning(f"Large state norm in {name}: {stats[f'{name}_norm']:.2f}")
                if torch.isnan(state).any() or torch.isinf(state).any():
                    logger.error(f"NaN/Inf in {name}!")
        
        self.state_history.append(stats)
        
        # Also record gate weights if available
        if hasattr(self.model, 'gate_history'):
            gate_vals = self.model.gate_history[:self.model.gate_ptr]
            if len(gate_vals) > 0:
                self.gate_history.append(gate_vals.mean(0).cpu().numpy())
        
        return stats

# ========== THE ACTUAL TRAINING LOOP ==========

class RealTrainingLoop:
    """
    Training loop with all the debugging and monitoring we actually need.
    """
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Debugging tools
        self.grad_debugger = GradientDebugger(model)
        self.state_monitor = StateMonitor(model)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )
        
        # Optimizer - CAREFULLY TUNED
        self.optimizer = self.create_optimizer()
        
        # Scheduler - CRITICAL FOR SSMs
        self.scheduler = self.create_scheduler()
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get('use_amp', True))
        
        # Tracking
        self.step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        import os
        os.makedirs('checkpoints', exist_ok=True)
        
        logger.info("Initialized training loop")
        log_gpu_memory()
    
    def create_optimizer(self):
        """Create optimizer with different settings for different components"""
        
        # Separate parameters by type
        ssm_params = []
        rwkv_params = []
        norm_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'ssm' in name.lower():
                ssm_params.append(param)
            elif 'rwkv' in name.lower() or 'memory' in name.lower():
                rwkv_params.append(param)
            elif 'norm' in name.lower() or 'ln' in name.lower():
                norm_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates for different components
        param_groups = [
            {'params': ssm_params, 'lr': self.config['lr'] * 0.5, 'weight_decay': 0.1},
            {'params': rwkv_params, 'lr': self.config['lr'] * 0.3, 'weight_decay': 0.01},
            {'params': norm_params, 'lr': self.config['lr'], 'weight_decay': 0.0},
            {'params': other_params, 'lr': self.config['lr'], 'weight_decay': 0.1}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        logger.info(f"Optimizer created with {len(ssm_params)} SSM params, {len(rwkv_params)} RWKV params")
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        
        # Warmup then cosine decay
        def lr_lambda(step):
            # Warmup
            if step < self.config.get('warmup_steps', 1000):
                return float(step) / float(max(1, self.config['warmup_steps']))
            
            # Cosine decay
            progress = float(step - self.config['warmup_steps']) / float(max(
                1, self.config['total_steps'] - self.config['warmup_steps']
            ))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return scheduler
    
    def compute_loss(self, logits, targets, states, phase='train'):
        """Compute loss with stability penalties"""
        
        # Base cross-entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # State stability penalty (only during training)
        stability_loss = 0
        if phase == 'train':
            for name, state in states.items():
                if state is not None:
                    # Penalize states that are too large or too small
                    state_norm = torch.norm(state, dim=-1).mean()
                    target_norm = 5.0  # Reasonable target norm
                    
                    # Use MSE loss to encourage stable norms
                    stability_loss += F.mse_loss(
                        state_norm,
                        torch.tensor(target_norm, device=state.device)
                    )
        
        # Output consistency penalty
        # Encourage smooth outputs (prevent abrupt changes)
        if logits.size(1) > 1:
            output_diff = logits[:, 1:] - logits[:, :-1]
            consistency_loss = output_diff.norm(dim=-1).mean() * 0.01
        else:
            consistency_loss = 0
        
        # Combine losses
        total_loss = ce_loss + self.config.get('stability_weight', 0.1) * stability_loss + consistency_loss
        
        # Log components
        loss_dict = {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'stability_loss': stability_loss.item() if isinstance(stability_loss, torch.Tensor) else stability_loss,
            'consistency_loss': consistency_loss
        }
        
        return total_loss, loss_dict
    
    def train_step(self, batch):
        """Single training step with all the debugging"""
        
        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        
        # Mixed precision forward
        with torch.cuda.amp.autocast(enabled=self.config.get('use_amp', True)):
            logits, states = self.model(input_ids)
            loss, loss_dict = self.compute_loss(logits, target_ids, states, phase='train')
        
        # Backward with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping - ABSOLUTELY CRITICAL
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.get('clip_grad', 1.0),
            norm_type=2
        )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Step scheduler
        self.scheduler.step()
        
        # Record state statistics
        state_stats = self.state_monitor.record_states(states)
        
        # Update step counter
        self.step += 1
        
        return loss_dict, state_stats
    
    def validate(self):
        """Validation with long-context testing"""
        self.model.eval()
        
        val_losses = []
        val_state_stats = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # Forward pass
                logits, states = self.model(input_ids)
                loss, loss_dict = self.compute_loss(logits, target_ids, states, phase='val')
                
                val_losses.append(loss_dict['loss'])
                val_state_stats.append(self.state_monitor.record_states(states))
        
        # Compute average stats
        avg_loss = np.mean(val_losses)
        
        # Log validation results
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        
        # Check state statistics
        if val_state_stats:
            avg_state_norm = np.mean([s.get('block_0_ssm_norm', 0) for s in val_state_stats])
            logger.info(f"Average State Norm: {avg_state_norm:.4f}")
        
        self.model.train()
        return avg_loss
    
    def long_context_test(self, context_lengths=[1024, 4096, 8192, 16384]):
        """
        Test model stability on increasingly long contexts.
        This is the REAL test for SSMs.
        """
        self.model.eval()
        
        logger.info("=" * 50)
        logger.info("LONG CONTEXT STABILITY TEST")
        logger.info("=" * 50)
        
        results = {}
        
        with torch.no_grad():
            for ctx_len in context_lengths:
                logger.info(f"Testing context length: {ctx_len}")
                
                # Generate a long sequence
                test_input = torch.randint(
                    0, self.model.vocab_size,
                    (2, ctx_len),  # Batch size 2
                    device=self.device
                )
                
                # Time the forward pass
                import time
                start_time = time.time()
                
                logits, states = self.model(test_input)
                
                inference_time = time.time() - start_time
                
                # Check memory usage
                log_gpu_memory()
                
                # Check perplexity (rough estimate)
                target = test_input[:, 1:]
                logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
                target_flat = target.reshape(-1)
                
                ce_loss = F.cross_entropy(logits_flat, target_flat)
                perplexity = torch.exp(ce_loss).item()
                
                # Check state stability
                state_norms = []
                for name, state in states.items():
                    if state is not None:
                        norm = torch.norm(state, dim=-1).mean().item()
                        state_norms.append(norm)
                
                avg_state_norm = np.mean(state_norms) if state_norms else 0
                
                results[ctx_len] = {
                    'perplexity': perplexity,
                    'inference_time': inference_time,
                    'avg_state_norm': avg_state_norm,
                    'tokens_per_second': ctx_len / inference_time if inference_time > 0 else 0
                }
                
                logger.info(f"  Perplexity: {perplexity:.2f}")
                logger.info(f"  Avg State Norm: {avg_state_norm:.2f}")
                logger.info(f"  Tokens/sec: {ctx_len / inference_time:.0f}")
                logger.info(f"  Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                
                # Reset states for next test
                if hasattr(self.model, 'reset_states'):
                    self.model.reset_states()
        
        self.model.train()
        return results
    
    def save_checkpoint(self, name, metrics=None):
        """Save checkpoint with all necessary state"""
        
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'metrics': metrics or {}
        }
        
        # Save checkpoint
        path = f"checkpoints/{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
        # Also save config separately
        with open(f"checkpoints/{name}_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_checkpoint(self, path):
        """Load checkpoint"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.step = checkpoint['step']
        
        logger.info(f"Loaded checkpoint from step {self.step}")
    
    def train(self, epochs=10, save_every=1000, validate_every=500):
        """
        Main training loop with all debugging and monitoring.
        """
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Total steps: {len(self.train_loader) * epochs}")
        
        # Initial validation
        initial_val_loss = self.validate()
        self.best_val_loss = initial_val_loss
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Progress bar
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch in pbar:
                # Train step
                loss_dict, state_stats = self.train_step(batch)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['loss']:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}",
                    'step': self.step
                })
                
                # Log to wandb (optional)
                if self.config.get('use_wandb', False):
                    wandb.log({
                        'train/loss': loss_dict['loss'],
                        'train/ce_loss': loss_dict['ce_loss'],
                        'train/stability_loss': loss_dict['stability_loss'],
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'step': self.step
                    })
                
                # Save checkpoint
                if self.step % save_every == 0:
                    self.save_checkpoint(f"step_{self.step}", metrics=loss_dict)
                
                # Validate
                if self.step % validate_every == 0:
                    val_loss = self.validate()
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model", metrics={'val_loss': val_loss})
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    
                    # Early stopping
                    if self.patience_counter >= self.config.get('patience', 10):
                        logger.info("Early stopping triggered")
                        return
                
                # Long context test (less frequent)
                if self.step % (validate_every * 5) == 0:
                    self.long_context_test()
            
            # End of epoch
            logger.info(f"End of epoch {epoch + 1}")
            log_gpu_memory()
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        logger.info("Training completed!")

# ========== CONFIGURATION ==========

def get_config():
    """Configuration that actually works"""
    
    return {
        # Model
        'vocab_size': 10000,
        'dim': 512,           # Start small
        'depth': 8,           # Shallow
        'ssm_dim': 16,        # Tiny state
        'rwkv_heads': 4,      # Few heads
        
        # Training
        'batch_size': 8,      # Small batches for stability
        'lr': 3e-4,           # Learning rate
        'warmup_steps': 1000,
        'total_steps': 50000,
        'clip_grad': 1.0,     # ABSOLUTELY NECESSARY
        
        # Dataset
        'seq_len': 1024,      # Start with short sequences
        'stride': 256,
        
        # Monitoring
        'use_amp': True,      # Mixed precision
        'use_wandb': False,   # Set to True if using wandb
        
        # Stability
        'stability_weight': 0.1,  # Weight for stability loss
        
        # Early stopping
        'patience': 10,
    }

# ========== MAIN TRAINING SCRIPT ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', nargs='+', required=True, help='Training files')
    parser.add_argument('--val_files', nargs='+', required=True, help='Validation files')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    args = parser.parse_args()
    
    # Get config
    config = get_config()
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TextDataset(args.train_files, seq_len=config['seq_len'])
    val_dataset = TextDataset(args.val_files, seq_len=config['seq_len'])
    
    # Update vocab size
    config['vocab_size'] = train_dataset.vocab_size
    
    # Create model
    logger.info("Creating model...")
    model = GroundedMamba(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        depth=config['depth'],
        ssm_dim=config['ssm_dim'],
        rwkv_heads=config['rwkv_heads']
    )
    
    # Create training loop
    trainer = RealTrainingLoop(model, train_dataset, val_dataset, config)
    
    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    trainer.train(
        epochs=10,
        save_every=1000,
        validate_every=500
    )