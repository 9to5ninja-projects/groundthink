# resilient_1b_colab.py
# INTELLIGENT HYBRID: RESILIENT 1B TRAINING SCRIPT (COLAB T4 OPTIMIZED)
# Based on Gemini 3 "Resilient 1B" Design

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import bitsandbytes as bnb

# Handle environment (Colab vs Local)
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("⚠️ Not running in Colab - Google Drive features disabled")

# ==========================================
# 1. CONFIGURATION (1B Scale / T4 Optimization)
# ==========================================
class Config:
    vocab_size = 50257   # GPT2 Tokenizer standard
    d_model = 2048       # 1B Scale hidden dim
    n_layer = 18         # Depth
    head_size = 64       # Tensor Core friendly
    grad_accum_steps = 16 # Simulates large batch
    micro_batch_size = 4  # Fits in T4 VRAM
    learning_rate = 4e-4
    max_seq_len = 512     # TinyStories context
    project_name = "groundthink_1B"

# ==========================================
# 2. THE SELECTIVE-WKV BLOCK (Matrix State)
# ==========================================
class SelectiveWKV_1B(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.d_model
        self.n_head = config.d_model // config.head_size
        self.head_size = config.head_size
        
        # Projections for Selective Gates (Mamba style)
        self.x_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.w_proj = nn.Linear(self.dim, self.dim) # Selective Decay
        
        # RWKV-style Key/Value/Receptance
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.r_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        
        # Layer Norms
        self.ln_x = nn.LayerNorm(self.dim)

    def forward(self, x, state=None):
        B, T, C = x.size()
        x = self.ln_x(x)
        
        # Selection logic
        # w: (B, T, C) -> Decides what to forget/remember per channel
        w = torch.sigmoid(self.w_proj(self.x_proj(x))) 
        
        k = self.k_proj(x)
        v = self.v_proj(x)
        r = torch.sigmoid(self.r_proj(x))
        
        # Reshape for head-wise matrix updates
        # To get Matrix State S = Sum(k * v^T), we need k=col, v=row
        # k: [B, T, H, D, 1]
        # v: [B, T, H, 1, D]
        # k @ v -> [D, D]
        k = k.view(B, T, self.n_head, self.head_size, 1)
        v = v.view(B, T, self.n_head, 1, self.head_size)
        w = w.view(B, T, self.n_head, self.head_size, 1) # Decay per Key-dimension
        
        # Initial State: [B, n_head, head_size, head_size]
        if state is None:
            state = torch.zeros(B, self.n_head, self.head_size, self.head_size, device=x.device)
        
        # The Grounded Recurrence (Python Loop for 1.0 logic, Triton needed for speed later)
        outputs = []
        for t in range(T):
            # Matrix update: S = (1-w)*S + (k*v)
            # k=[D,1], v=[1,D] -> kv=[D,D] (Outer Product)
            kv = k[:, t] @ v[:, t] 
            
            # Weighted decay + Update
            state = (1 - w[:, t]) * state + kv
            
            # Read gate (Receptance)
            # r_t: [B, H, 1, D] -> Query vector (row)
            r_t = r[:, t].view(B, self.n_head, 1, self.head_size)
            
            # Output: r @ state
            # [1, D] @ [D, D] -> [1, D]
            # (r @ k) * v -> scalar * v -> v (weighted)
            context = r_t @ state
            outputs.append(context.view(B, C))
            
        return self.out_proj(torch.stack(outputs, dim=1)), state

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

        # Weight tying
        self.token_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

# ==========================================
# 3. DATA & TOKENIZATION
# ==========================================
def get_dataloaders(config):
    print("📚 Loading TinyStories dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=config.max_seq_len,
            return_tensors="pt"
        )
    
    # Custom collate for streaming dataset
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encoded = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=config.max_seq_len, 
            return_tensors="pt"
        )
        input_ids = encoded['input_ids']
        labels = input_ids.clone()
        return input_ids, labels

    return DataLoader(dataset, batch_size=config.micro_batch_size, collate_fn=collate_fn)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    config = Config()
    
    # Setup Drive
    save_dir = f"checkpoints/{config.project_name}"
    if IN_COLAB:
        drive.mount('/content/drive')
        save_dir = f"/content/drive/MyDrive/{config.project_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Model Setup
    print(f"🚀 Initializing GroundThink-1B ({config.d_model} dim, {config.n_layer} layers)")
    model = GroundThink1B(config).cuda()
    
    # 8-bit Adam for T4 efficiency
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.learning_rate)
    
    dataloader = get_dataloaders(config)
    
    # Resume?
    start_step = 0
    checkpoint_path = os.path.join(save_dir, "latest.pt")
    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        start_step = ckpt['step']
    
    print("🔥 Starting Training...")
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0
    t0 = time.time()
    
    for step, (x, y) in enumerate(dataloader, start=start_step):
        x, y = x.cuda(), y.cuda()
        
        # Forward
        _, loss = model(x, y)
        loss = loss / config.grad_accum_steps
        loss.backward()
        
        running_loss += loss.item() * config.grad_accum_steps
        
        # Step (Gradient Accumulation)
        if (step + 1) % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging
            if (step + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"Step {step+1} | Loss: {running_loss/config.grad_accum_steps:.4f} | Time: {dt:.2f}s")
                running_loss = 0
                t0 = time.time()
        
        # Checkpoint (Resilient)
        if (step + 1) % 500 == 0:
            print(f"💾 Saving checkpoint at step {step+1}...")
            torch.save({
                'step': step + 1,
                'model': model.state_dict(),
                'opt': optimizer.state_dict()
            }, checkpoint_path)

if __name__ == "__main__":
    train()
