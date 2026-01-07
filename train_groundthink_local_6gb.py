import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
import bitsandbytes as bnb

# -----------------------------------------------------------------------------
# 0. LOCAL HARDWARE CONFIGURATION
# -----------------------------------------------------------------------------
# Check if we can use TF32 (RTX 30xx/40xx) or need standard FP32 math
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')
    print("🚀 Ampere+ GPU detected. TF32 Enabled.")
else:
    print("ℹ️ Older GPU detected. Using standard precision.")

# Import Triton (Speedup)
try:
    # Try importing assuming script is in root and groundthink is a package
    from groundthink.ops.selective_scan_triton import selective_scan_triton_forward
    USE_TRITON = True
    print("✅ Triton Kernel available. Acceleration ON.")
except ImportError:
    try:
        # Try relative import if script is moved or path differs
        from ops.selective_scan_triton import selective_scan_triton_forward
        USE_TRITON = True
        print("✅ Triton Kernel available (Relative). Acceleration ON.")
    except ImportError as e:
        USE_TRITON = False
        print(f"⚠️ Triton not found: {e}. Using (slow) Python fallback.")

# -----------------------------------------------------------------------------
# 1. TRAIN CONFIG (TUNED FOR 6GB VRAM)
# -----------------------------------------------------------------------------
class Config:
    project_name = "groundthink_1B_local"
    
    # Model Architecture (Unchanged - Real 1B Scale)
    vocab_size = 50257
    d_model = 2048          # Full 1B width
    n_layer = 24            # Full 1B depth
    head_size = 64
    
    # LOCAL OPTIMIZATION
    # Strategy: Trade VRAM for Compute/System RAM
    # ---------------------------------------------------------------
    micro_batch_size = 1    # Process 1 sample at a time (Limit VRAM usage)
    grad_accum_steps = 128  # 1 * 128 = Effective Batch Size 128 (Stable training)
    
    max_seq_len = 1024      # Reduced from 2048 to ensure 1B fits in 6GB. 
                            # (Can try 2048 if 1024 is stable)
    
    total_steps = 2500      # Stop after this many steps
                            
    learning_rate = 3e-4
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

config = Config()

# -----------------------------------------------------------------------------
# 2. MODEL DEFINITION (Same Architecture)
# -----------------------------------------------------------------------------
class SelectiveWKV_1B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.d_model
        self.n_head = config.d_model // config.head_size
        self.head_size = config.head_size
        
        self.x_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.w_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.r_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.ln_x = nn.LayerNorm(self.dim)

    def forward(self, x, state=None):
        B, T, C = x.size()
        x = self.ln_x(x)
        
        # Projections
        w = torch.sigmoid(self.w_proj(self.x_proj(x))) 
        k = self.k_proj(x)
        v = self.v_proj(x)
        r = torch.sigmoid(self.r_proj(x))
        
        # Reshape [B, T, H, D]
        k = k.view(B, T, self.n_head, self.head_size) 
        v = v.view(B, T, self.n_head, self.head_size) 
        w = w.view(B, T, self.n_head, self.head_size) 
        r = r.view(B, T, self.n_head, self.head_size)
        
        if USE_TRITON and x.is_cuda:
            y = selective_scan_triton_forward(k, v, w, r, state)
            y = y.view(B, T, C)
            return self.out_proj(y), None
            
        else:
            # Fallback for CPU testing
            k = k.unsqueeze(-1); v = v.unsqueeze(-2); w = w.unsqueeze(-1); r = r.unsqueeze(-2)
            if state is None: 
                state = torch.zeros(B, self.n_head, self.head_size, self.head_size, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                kv = k[:, t] @ v[:, t]
                state = (1 - w[:, t]) * state + kv
                # Ensure state matches r dtype for matmul if they drifted
                out = r[:, t] @ state.to(r.dtype) 
                outs.append(out.squeeze(2))
            return self.out_proj(torch.stack(outs, dim=1).view(B, T, C)), state

class GroundThinkBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.mixer = SelectiveWKV_1B(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model)
        )

    def forward(self, x):
        # Gradient Checkpointing is MANDATORY for 6GB VRAM
        # It prevents storing intermediate activations for every layer
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        # Self-Attention/Mixer part
        # We manually checkpoint the heavy parts
        mixer_out, _ = self.mixer(self.ln1(x))
        x = x + mixer_out
        
        # MLP part
        x = x + self.mlp(self.ln2(x))
        return x

class GroundThink1B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([GroundThinkBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight # Weight tying

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        
        # Aggressive Checkpointing
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
            
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return self.head(x), None

# -----------------------------------------------------------------------------
# 3. COMPRESSED TRAINING LOOP
# -----------------------------------------------------------------------------
def get_dataloaders(config):
    print("📚 Loading Dataset (Streaming Mode)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encoded = tokenizer(
            texts, padding='max_length', truncation=True, 
            max_length=config.max_seq_len, return_tensors="pt"
        )
        
        input_ids = encoded['input_ids']
        labels = input_ids.clone()
        
        # Mask padding tokens so we don't train on them
        if 'attention_mask' in encoded:
            labels[encoded['attention_mask'] == 0] = -100
            
        return input_ids, labels
        
    return DataLoader(dataset, batch_size=config.micro_batch_size, collate_fn=collate_fn)

def train_local():
    print(f"\n🧠 Initializing GroundThink 1B (Local Mode)")
    print(f"   VRAM Target: < 6GB | Method: Paged Optim + Grad Checkpoint")
    print(f"   Batch: {config.micro_batch_size} (Micro) -> {config.micro_batch_size*config.grad_accum_steps} (Effective)")
    
    model = GroundThink1B(config)
    
    # Enable BF16/FP16 native
    model = model.to(dtype=config.dtype, device='cuda')
    
    print(f"   Model Parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    
    # CRITICAL: Use Paged AdamW 8-bit
    # This stores optimizer state in SYSTEM RAM, not VRAM.
    print("   Optimizer: Paged 8-bit AdamW (Saving ~6GB VRAM)")
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=config.learning_rate)
    
    dataloader = get_dataloaders(config)
    model.train()
    
    step_loss = 0
    accum_counter = 0
    t0 = time.time()
    
    print("\n🚀 Starting Training Loop...")
    
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to('cuda'), y.to('cuda')
        
        # Forward
        logits, loss = model(x, y)
        
        # Scale loss for accumulation
        loss = loss / config.grad_accum_steps
        step_loss += loss.item()
        
        # Backward (Accumulate gradients)
        loss.backward()
        
        accum_counter += 1
        
        # Update weights only when accumulation is full
        if accum_counter % config.grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Stats
            dt = time.time() - t0
            effective_tokens = config.micro_batch_size * config.max_seq_len * config.grad_accum_steps
            tps = effective_tokens / dt
            
            print(f"Step {accum_counter // config.grad_accum_steps} | Loss: {step_loss:.4f} | Speed: {tps:.0f} tok/s")
            
            # SAVE CHECKPOINT
            current_step = accum_counter // config.grad_accum_steps
            if current_step % 50 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/step_{current_step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"💾 Checkpoint saved: {ckpt_path}")

            # Stop condition
            if current_step >= config.total_steps:
                print(f"🏁 Training complete! Reached {current_step} steps.")
                final_path = f"checkpoints/final_model.pth"
                torch.save(model.state_dict(), final_path)
                print(f"💾 Final model saved: {final_path}")
                break

            step_loss = 0
            t0 = time.time()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ This script requires a NVIDIA GPU (even a small one).")
    else:
        # Check VRAM
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"ℹ️  Detected GPU VRAM: {vram:.2f} GB")
        train_local()
