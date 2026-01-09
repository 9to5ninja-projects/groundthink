"""
Train GroundedMamba on small dataset
Per FOUNDATION.md: validate on small dataset before scaling
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging

from groundthink.grounded_model import GroundedMamba
from groundthink.grounded_training import (
    create_optimizer, get_scheduler, compute_loss,
    GradientDebugger, StateMonitor, EmergencyRecovery, check_loss_health
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleTextDataset(Dataset):
    """Character-level dataset for validation"""
    
    def __init__(self, text: str, seq_len: int = 256):
        self.seq_len = seq_len
        
        # Build vocab
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Tokenize
        self.tokens = [self.stoi[ch] for ch in text]
        logger.info(f"Dataset: {len(self.tokens)} tokens, vocab size: {self.vocab_size}")
    
    def __len__(self):
        return max(1, len(self.tokens) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train_small():
    """Train on small synthetic dataset"""
    
    # Create simple repeating text for validation
    # Model should learn to predict patterns
    text = "Hello world! " * 1000
    text += "The quick brown fox jumps over the lazy dog. " * 500
    text += "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 200
    
    dataset = SimpleTextDataset(text, seq_len=128)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model with dataset's vocab size
    model = GroundedMamba(
        vocab_size=dataset.vocab_size,
        dim=256,  # Even smaller for quick validation
        depth=4,
        ssm_dim=8,
        rwkv_heads=2,
    )
    logger.info(f"Model params: {model.n_params:,}")
    
    # Training setup
    optimizer = create_optimizer(model, lr=1e-3)
    scheduler = get_scheduler(optimizer, warmup_steps=100, total_steps=1000)
    
    # Debugging tools
    grad_debugger = GradientDebugger(model)
    state_monitor = StateMonitor()
    recovery = EmergencyRecovery(model, optimizer)
    
    # Training loop
    model.train()
    step = 0
    best_loss = float('inf')
    
    for epoch in range(5):
        epoch_losses = []
        
        for batch_x, batch_y in loader:
            step += 1
            
            # Forward
            logits, states = model(batch_x)
            loss, loss_dict = compute_loss(logits, batch_y, states)
            
            # Check for NaN/Inf
            if not check_loss_health(loss, optimizer):
                recovery.recover()
                continue
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            grad_debugger.check_and_fix()
            
            # Clip gradients (NON-NEGOTIABLE per FOUNDATION.md)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss_dict['total_loss'])
            
            # Log every 50 steps
            if step % 50 == 0:
                avg_loss = sum(epoch_losses[-50:]) / min(50, len(epoch_losses))
                lr = optimizer.param_groups[0]['lr']
                logger.info(f"Step {step}: loss={avg_loss:.4f}, lr={lr:.6f}")
                
                # Check state norms
                stats, warnings = state_monitor.check(states)
                for w in warnings:
                    logger.warning(w)
                
                # Save good state
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    recovery.save_good_state(step)
            
            if step >= 500:
                break
        
        if step >= 500:
            break
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch+1}: avg_loss={avg_epoch_loss:.4f}")
    
    # Final validation
    logger.info("=" * 50)
    logger.info("Training complete. Final validation:")
    
    model.eval()
    with torch.no_grad():
        # Test on a sample
        test_text = "Hello "
        test_tokens = [dataset.stoi.get(c, 0) for c in test_text]
        x = torch.tensor([test_tokens], dtype=torch.long)
        
        logits, _ = model(x)
        probs = F.softmax(logits[0, -1], dim=-1)
        top_idx = probs.topk(5).indices.tolist()
        top_chars = [dataset.itos[i] for i in top_idx]
        
        logger.info(f"Input: '{test_text}'")
        logger.info(f"Top predictions: {top_chars}")
    
    logger.info(f"Best loss achieved: {best_loss:.4f}")
    logger.info("✓ Small dataset validation complete")


if __name__ == "__main__":
    train_small()
