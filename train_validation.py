"""
TRAINING VALIDATION FOR INTELLIGENT HYBRID
Step 1: Validate that the model can learn on a small dataset

Based on the original design documents with proper:
- Gradient clipping (NON-NEGOTIABLE for SSMs)
- Weight decay (0.1 strict)
- Learning rate warmup
- State monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time
import os
import sys

sys.path.insert(0, '.')
from intelligent_hybrid import IntelligentHybridModel, INTELLIGENT_SCALING

# ========== SIMPLE TEXT DATASET ==========

class SimpleTextDataset(Dataset):
    """
    Simple character-level dataset for validation training.
    We just need to verify the model can learn - not train a good model.
    """
    
    def __init__(self, text, seq_len=256, stride=128):
        self.seq_len = seq_len
        self.stride = stride
        
        # Build vocabulary
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Tokenize
        self.tokens = [self.stoi[ch] for ch in text]
        
        print(f"Dataset: {len(self.tokens)} tokens, vocab size: {self.vocab_size}")
    
    def __len__(self):
        return max(1, (len(self.tokens) - self.seq_len - 1) // self.stride)
    
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len + 1
        
        tokens = self.tokens[start:end]
        
        # Pad if necessary
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y


# ========== TRAINING CONFIG ==========

TRAIN_CONFIG = {
    # From design doc
    'learning_rate': 3e-4,
    'weight_decay': 0.1,  # Strict - prevents parameter drift
    'gradient_clip': 1.0,  # NON-NEGOTIABLE for SSMs
    'warmup_steps': 100,
    
    # Training
    'batch_size': 4,
    'seq_len': 256,
    'epochs': 5,
    
    # Monitoring
    'log_interval': 50,
    'eval_interval': 200,
}


# ========== TRAINING LOOP ==========

class ValidationTrainer:
    """
    Simple trainer to validate model can learn.
    """
    
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # Optimizer with weight decay (from design doc)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # Tracking
        self.step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def get_lr(self):
        """Learning rate with warmup"""
        if self.step < self.config['warmup_steps']:
            return self.config['learning_rate'] * (self.step / self.config['warmup_steps'])
        return self.config['learning_rate']
    
    def train_step(self, x, y):
        """Single training step"""
        self.model.train()
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr()
        
        # Forward
        x, y = x.to(self.device), y.to(self.device)
        logits, states = self.model(x)
        
        # Loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (NON-NEGOTIABLE for SSMs)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config['gradient_clip']
        )
        
        # Check for gradient issues
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"⚠️ Bad gradient at step {self.step}: {grad_norm}")
            self.optimizer.zero_grad()
            return None, None
        
        self.optimizer.step()
        self.step += 1
        
        return loss.item(), grad_norm.item()
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("VALIDATION TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Weight decay: {self.config['weight_decay']}")
        print(f"Gradient clip: {self.config['gradient_clip']}")
        
        # Initial evaluation
        val_loss = self.evaluate()
        print(f"\nInitial val loss: {val_loss:.4f} (perplexity: {math.exp(val_loss):.2f})")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{self.config['epochs']} ---")
            
            epoch_losses = []
            
            for batch_idx, (x, y) in enumerate(self.train_loader):
                loss, grad_norm = self.train_step(x, y)
                
                if loss is None:
                    continue
                
                epoch_losses.append(loss)
                self.train_losses.append(loss)
                
                # Logging
                if self.step % self.config['log_interval'] == 0:
                    avg_loss = sum(epoch_losses[-50:]) / len(epoch_losses[-50:])
                    lr = self.get_lr()
                    elapsed = time.time() - start_time
                    
                    print(f"  Step {self.step:5d} | "
                          f"Loss: {loss:.4f} | "
                          f"Avg: {avg_loss:.4f} | "
                          f"Grad: {grad_norm:.3f} | "
                          f"LR: {lr:.2e} | "
                          f"Time: {elapsed:.1f}s")
                
                # Evaluation
                if self.step % self.config['eval_interval'] == 0:
                    val_loss = self.evaluate()
                    self.val_losses.append((self.step, val_loss))
                    
                    improved = val_loss < self.best_val_loss
                    if improved:
                        self.best_val_loss = val_loss
                    
                    print(f"  📊 Val loss: {val_loss:.4f} "
                          f"(ppl: {math.exp(val_loss):.2f}) "
                          f"{'⬇️ NEW BEST' if improved else ''}")
            
            # End of epoch summary
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"  Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}")
        
        # Final evaluation
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        final_val_loss = self.evaluate()
        initial_val_loss = self.val_losses[0][1] if self.val_losses else float('inf')
        
        print(f"Initial val loss: {initial_val_loss:.4f}")
        print(f"Final val loss: {final_val_loss:.4f}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Improvement: {initial_val_loss - final_val_loss:.4f}")
        
        # Check if model learned
        learned = final_val_loss < initial_val_loss * 0.9  # At least 10% improvement
        
        if learned:
            print("\n✅ MODEL IS LEARNING - Architecture validated")
        else:
            print("\n⚠️ MODEL NOT LEARNING WELL - May need adjustment")
        
        return {
            'initial_loss': initial_val_loss,
            'final_loss': final_val_loss,
            'best_loss': self.best_val_loss,
            'learned': learned,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model': self.model,  # RETURN THE TRAINED MODEL
        }


# ========== GENERATE SAMPLE TEXT ==========

def generate_sample_text(num_chars=50000):
    """
    Generate simple text for training validation.
    Uses patterns that are easy to learn but non-trivial.
    """
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "Hello world! This is a test of the hybrid model. ",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ. ",
        "0123456789. ",
        "The cat sat on the mat. The dog ran in the park. ",
        "Once upon a time, in a land far away, there lived a wise old owl. ",
        "To be or not to be, that is the question. ",
        "All that glitters is not gold. ",
    ]
    
    text = ""
    while len(text) < num_chars:
        for pattern in patterns:
            text += pattern
            if len(text) >= num_chars:
                break
    
    return text[:num_chars]


# ========== MAIN ==========

if __name__ == "__main__":
    print("🚀 INTELLIGENT HYBRID - VALIDATION TRAINING")
    print("="*60)
    
    # Step 1: Create model
    print("\n1. Creating model (valid scale - ~15M params)")
    config = INTELLIGENT_SCALING['valid']
    
    # Generate sample text
    print("\n2. Generating training data...")
    train_text = generate_sample_text(100000)  # 100k chars for training
    val_text = generate_sample_text(10000)     # 10k chars for validation
    
    # Create datasets
    train_dataset = SimpleTextDataset(train_text, seq_len=256, stride=128)
    val_dataset = SimpleTextDataset(val_text, seq_len=256, stride=256)
    
    # Create model with matching vocab size
    model = IntelligentHybridModel(
        vocab_size=train_dataset.vocab_size,
        dim=config['dim'],
        depth=config['depth'],
        state_dim=config['state_dim']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Vocab size: {train_dataset.vocab_size}")
    
    # Step 2: Train
    print("\n3. Starting validation training...")
    trainer = ValidationTrainer(model, train_dataset, val_dataset, TRAIN_CONFIG)
    results = trainer.train()
    
    # Step 3: Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if results['learned']:
        print("✅ ARCHITECTURE LEARNS SUCCESSFULLY")
        
        # ========== CONTEXT RETENTION TEST ==========
        print("\n" + "="*60)
        print("CONTEXT RETENTION TEST (trained model)")
        print("="*60)
        
        # Get the TRAINED model from results
        trained_model = results['model']
        trained_model.eval()
        device = next(trained_model.parameters()).device
        vocab_size = train_dataset.vocab_size
        
        print(f"   Vocab size: {vocab_size}")
        print(f"   Testing context influence on trained model...")
        
        with torch.no_grad():
            # Test: Do early tokens influence late outputs?
            # Use VALID token indices (0 to vocab_size-1)
            max_token = min(50, vocab_size - 1)
            
            # Two sequences that differ ONLY at the start
            base_seq = list(range(10, min(60, vocab_size)))  # tokens 10-59 or less
            seq_len = len(base_seq)
            
            # Ensure we have enough tokens
            if seq_len < 20:
                print("   ⚠️ Vocab too small for meaningful test")
            else:
                # Sequence A: starts with [0,1,2,3,4]
                # Sequence B: starts with [5,6,7,8,9]
                seq_a = [0, 1, 2, 3, 4] + base_seq[5:]
                seq_b = [5, 6, 7, 8, 9] + base_seq[5:]
                
                x_a = torch.tensor([seq_a], device=device)
                x_b = torch.tensor([seq_b], device=device)
                
                out_a = trained_model(x_a)
                out_b = trained_model(x_b)
                logits_a = out_a[0] if isinstance(out_a, tuple) else out_a
                logits_b = out_b[0] if isinstance(out_b, tuple) else out_b
                
                # Check logits at END of sequence - do they differ based on START?
                end_logits_a = logits_a[0, -5:]  # Last 5 positions
                end_logits_b = logits_b[0, -5:]
                
                diff = (end_logits_a - end_logits_b).abs().mean().item()
                
                print(f"   Sequence length: {len(seq_a)}")
                print(f"   Sequences differ only at positions 0-4")
                print(f"   End logits difference: {diff:.6f}")
                
                if diff > 0.1:
                    print("   ✅ Context retention: STRONG (early tokens influence late outputs)")
                elif diff > 0.01:
                    print("   ⚠️ Context retention: MODERATE")
                elif diff > 0.001:
                    print("   ⚠️ Context retention: WEAK")
                else:
                    print("   ❌ Context retention: NONE (early tokens have no influence)")
                
                # Also test longer range
                print("\n   Testing at different sequence lengths...")
                for test_len in [20, 50, 100, 200]:
                    if test_len > vocab_size:
                        continue
                    base = list(range(10, min(10 + test_len, vocab_size)))
                    if len(base) < test_len:
                        # Repeat tokens to reach length
                        while len(base) < test_len:
                            base.extend(range(10, min(10 + test_len - len(base), vocab_size)))
                    base = base[:test_len]
                    
                    sa = [0, 1, 2, 3, 4] + base[5:]
                    sb = [5, 6, 7, 8, 9] + base[5:]
                    
                    xa = torch.tensor([sa], device=device)
                    xb = torch.tensor([sb], device=device)
                    
                    oa = trained_model(xa)
                    ob = trained_model(xb)
                    la = oa[0] if isinstance(oa, tuple) else oa
                    lb = ob[0] if isinstance(ob, tuple) else ob
                    
                    d = (la[0, -1] - lb[0, -1]).abs().mean().item()
                    status = "✅" if d > 0.01 else "⚠️" if d > 0.001 else "❌"
                    print(f"     Len {test_len:4d}: diff={d:.6f} {status}")
        
        # Save the trained model
        save_path = "trained_valid_model.pt"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'vocab_size': vocab_size,
            'config': config,
            'stoi': train_dataset.stoi,
            'itos': train_dataset.itos,
        }, save_path)
        print(f"\n   💾 Model saved to: {save_path}")
        
        print("\n🎯 NEXT STEPS:")
        print("   1. Scale to 'small' (~30M) and repeat")
        print("   2. Train on real data")
        print("\n" + "="*60)
        print("🎯 ALL VALIDATION COMPLETE - Ready to scale!")
        print("="*60)
    else:
        print("❌ LEARNING ISSUES DETECTED")
        print("   Review training dynamics and adjust")
