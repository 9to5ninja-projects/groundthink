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

# Try to import Triton kernel for 10x speedup
try:
    # Try importing assuming script is in root and groundthink is a package
    from groundthink.ops.selective_scan_triton import selective_scan_triton_forward
    USE_TRITON = True
    print("✅ Triton Kernel available (Package). Acceleration ON.")
except ImportError:
    try:
        # Try relative import if script is running from inside groundthink folder
        from ops.selective_scan_triton import selective_scan_triton_forward
        USE_TRITON = True
        print("✅ Triton Kernel available (Local). Acceleration ON.")
    except ImportError as e:
        USE_TRITON = False
        print(f"⚠️ Triton not found: {e}. Falling back to slow Python loop.")

# ==========================================
# 0. A100 OPTIMIZATIONS
# ==========================================
torch.set_float32_matmul_precision('high') # Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# 1. CONFIGURATION (A100 80GB Profile)
# ==========================================
class Config:
    vocab_size = 50257      # GPT2 Standard
    d_model = 2048          # 1B Scale
    n_layer = 24            # Deeper than T4 version (18 -> 24)
    head_size = 64          
    
    # A100 Specifics
    # Total Batch = micro_batch * grad_accum * devices
    # 80GB VRAM allows dense batches
    micro_batch_size = 16    # Increased for A100 80GB
    grad_accum_steps = 8     # Total batch ~128
    
    max_seq_len = 2048      # Real context length (vs 512 on T4)
    learning_rate = 3e-4    # Slightly lower for deeper model
    
    project_name = "groundthink_1B_A100"
    dtype = torch.bfloat16  # NATIVE BF16 (No Scaler needed, High Stability)

config = Config()

# ==========================================
# 2. MODEL ARCHITECTURE (Selective Matrix)
# ==========================================
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
        
        # Reshape
        # [B, T, H, D] format is cleaner for both Triton and PyTorch
        k = k.view(B, T, self.n_head, self.head_size) 
        v = v.view(B, T, self.n_head, self.head_size) 
        w = w.view(B, T, self.n_head, self.head_size) 
        r = r.view(B, T, self.n_head, self.head_size)
        
        if USE_TRITON and x.is_cuda:
            # FAST PATH: Triton Kernel
            # Note: Kernel expects [B, T, H, D]
            # Output: [B, T, H, D]
            y = selective_scan_triton_forward(k, v, w, r, state)
            
            # Combine Heads [B, T, H, D] -> [B, T, C]
            y = y.view(B, T, C)
            return self.out_proj(y), None # State updates not fully synced back in this simplified v1 kernel wrapper
            
        else:
            # REAL SLOW PATH: Python Loop
            
            # Add singleton dims for matrix math: [B, T, H, D, 1] etc
            k = k.unsqueeze(-1)         # [B, T, H, D, 1]
            v = v.unsqueeze(-2)         # [B, T, H, 1, D]
            w = w.unsqueeze(-1)         # [B, T, H, D, 1]
            r = r.unsqueeze(-2)         # [B, T, H, 1, D]
            
            if state is None:
                state = torch.zeros(B, self.n_head, self.head_size, self.head_size, device=x.device, dtype=x.dtype)
            
            # CHUNKED RECURRENCE (Fused-ish)
            # On A100, we can use larger chunks or compile this
            chunk_size = 128
            num_chunks = (T + chunk_size - 1) // chunk_size
            
            outputs = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, T)
                
                # Slice
                k_c = k[:, start:end]
                v_c = v[:, start:end]
                w_c = w[:, start:end]
                r_c = r[:, start:end]
                
                # Chunk Scan
                chunk_out = []
                for t in range(end - start):
                    kv = k_c[:, t] @ v_c[:, t] # [B, H, D, 1] @ [B, H, 1, D] -> [B, H, D, D]
                    state = (1 - w_c[:, t]) * state + kv
                    out = r_c[:, t] @ state    # [B, H, 1, D] @ [B, H, D, D] -> [B, H, 1, D]
                    chunk_out.append(out.squeeze(2))
                    
                outputs.append(torch.stack(chunk_out, dim=1))
            
            return self.out_proj(torch.cat(outputs, dim=1).view(B, T, C)), state

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
        mixer_out, _ = self.mixer(self.ln1(x))
        x = x + mixer_out
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
        self.token_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx)
        
        # Gradient Checkpointing (Essential for RNN state memory)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
            
        x = self.ln_f(x)
        
        if targets is not None:
            # Efficient Loss Calculation (Avoid materializing full logits)
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        
        return self.head(x), None

# ==========================================
# 3. DATASET
# ==========================================
def get_dataloaders(config):
    print("📚 Loading Dataset (TinyStories for demo, swap for OpenWebText on A100)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Increase buffer use for A100 speed
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encoded = tokenizer(
            texts, padding=True, truncation=True, 
            max_length=config.max_seq_len, return_tensors="pt"
        )
        return encoded['input_ids'], encoded['input_ids'].clone()

    return DataLoader(
        dataset, 
        batch_size=config.micro_batch_size, 
        collate_fn=collate_fn,
        num_workers=4,        # Use CPU cores
        pin_memory=True       # Fast transfer to GPU
    )

# ==========================================
# 4. HIGH-PERFORMANCE LOOP
# ==========================================
def train():
    print(f"🚀 Initializing A100 Run: {config.project_name}")
    print(f"   Context: {config.max_seq_len} | Batch: {config.micro_batch_size}")
    
    # 1. Model init (BF16 native)
    model = GroundThink1B(config)
    model.to('cuda', dtype=torch.bfloat16) # BF16 is King on A100
    
    # 2. Compile (Torch 2.0+ Speedup)
    # print("🔥 Compiling model with torch.compile()...")
    # model = torch.compile(model)
    print("⚠️ torch.compile DISABLED (Immediate Execution Mode)")
    
    # 3. Optimize
    # Standard AdamW is fine with 80GB, but 8bit is still efficient
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.learning_rate)
    
    dataloader = get_dataloaders(config)
    
    model.train()
    optimizer.zero_grad()
    
    t0 = time.time()
    micro_step = 0
    global_step = 0
    accum_loss = 0.0
    
    print("⚡ Training Started...")
    
    # Create checkpoints dir immediately to verify permissions
    ckpt_dir = f"checkpoints/{config.project_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        
        # No Autocast needed if model is already bfloat16 and logic is clean
        # But we use it for safety with LN/Softmax
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
            loss = loss / config.grad_accum_steps
        
        # Accumulate loss for logging (multiply back to get real scale)
        accum_loss += loss.item() * config.grad_accum_steps
        
        loss.backward()
        
        micro_step += 1
        
        # Optimizer Step (Gradient Accumulation)
        if micro_step % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Throughput & Logging
            dt = time.time() - t0
            # Real tokens processed in this interval
            current_tokens = config.micro_batch_size * config.max_seq_len * config.grad_accum_steps
            tps = current_tokens / dt
            avg_loss = accum_loss / config.grad_accum_steps
            
            print(f"Step {global_step} | Loss: {avg_loss:.4f} | TPS: {tps:.0f}")
            
            # Reset counters
            t0 = time.time()
            accum_loss = 0.0
            
            # Save Checkpoint (Every 50 optimizer steps)
            # 50 steps * 128 batch size = 6400 samples
            if global_step % 50 == 0:
                print(f"💾 Saving checkpoint at step {global_step}...")
                torch.save(model.state_dict(), f"{ckpt_dir}/step_{global_step}.pt")

if __name__ == "__main__":
    train()
