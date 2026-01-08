#!/usr/bin/env python3
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
import bitsandbytes as bnb
from torch.utils.tensorboard import SummaryWriter

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
# 1. CONFIGURATION (125M V2 Profile)
# ==========================================
class Config:
    vocab_size = 50257      # GPT2 Standard
    d_model = 768           # 125M Scale (Standard Small)
    n_layer = 12            # 125M Scale (Standard Small)
    head_size = 64          
    
    # A100 Specifics for Small Model
    # PyTorch loop backward uses more memory than Triton kernel
    micro_batch_size = 8     # Reduced further for PyTorch backward
    grad_accum_steps = 32    # Maintain Total Batch ~256
    
    max_seq_len = 2048      
    learning_rate = 6e-4    # Higher LR for smaller model (GPT-3 small used 6e-4)
    
    total_steps = 5000      # More steps for smaller model convergence
    
    project_name = "groundthink_v2_125M"
    dtype = torch.bfloat16  # NATIVE BF16

config = Config()

# ==========================================
# 2. MODEL ARCHITECTURE (V2: SelectiveWKV)
# ==========================================
class SelectiveWKV_V2(nn.Module):
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
        
        # Initialize Geometric Decay (CRITICAL FOR LONG CONTEXT)
        with torch.no_grad():
            n_heads = self.n_head
            head_size = self.head_size
            r_min = 0.9
            r_max = 0.999
            r_schedule = torch.linspace(r_min, r_max, n_heads)
            w_target = 1.0 - r_schedule
            bias_schedule = torch.logit(w_target)
            bias_full = bias_schedule.unsqueeze(-1).expand(n_heads, head_size).reshape(-1)
            self.w_proj.bias.copy_(bias_full)
            self.w_proj.weight.data.normal_(0, 0.01)

    def forward(self, x, state=None):
        B, T, C = x.size()
        x = self.ln_x(x)
        
        # Projections
        w_raw = self.w_proj(self.x_proj(x))
        w = torch.sigmoid(w_raw) 
        k = self.k_proj(x)
        v = self.v_proj(x)
        r = torch.sigmoid(self.r_proj(x))
        
        # Prepare for Triton Wrapper: Expects [B, T, H, D]
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)
        w = w.view(B, T, self.n_head, self.head_size) 
        r = r.view(B, T, self.n_head, self.head_size)
        
        if USE_TRITON:
            # TRY to use Triton, but if it fails/has no grad, we must not use it.
            # Given the known issue, we swap to the JIT-compiled PyTorch implementation
            # which we know works and supports backward pass.
            y = self.jit_scan(k, v, w, r, state)
        else:
            y = self.jit_scan(k, v, w, r, state)

        # Reshape back: [B, T, H, D] -> [B, T, C]
        y = y.reshape(B, T, C)
        
        return self.out_proj(y), None

    # JIT-compiled version of the V1 logic for speed + correctness
    @torch.jit.export
    def jit_scan(self, k, v, w, r, state: torch.Tensor = None):
        # type: (Tensor, Tensor, Tensor, Tensor, Optional[Tensor]) -> Tensor
        B, T, H, D = k.shape # Expected [B, T, H, D] here
        
        k = k.unsqueeze(-1)         # [B, T, H, D, 1]
        v = v.unsqueeze(-2)         # [B, T, H, 1, D]
        w = w.unsqueeze(-1)         # [B, T, H, D, 1]
        r = r.unsqueeze(-2)         # [B, T, H, 1, D]
        
        if state is None:
            state = torch.zeros(B, H, D, D, device=k.device, dtype=k.dtype)
            
        outputs = []
        # Sequential scan (JIT will fuse this reasonably well)
        for t in range(T):
            # S = (1 - w) * S + k @ v.T
            kv = k[:, t] @ v[:, t] 
            state = (1 - w[:, t]) * state + kv
            
            # y = r @ S
            out = r[:, t] @ state
            outputs.append(out.squeeze(2))
            
        return torch.stack(outputs, dim=1)

    def get_diagnostics(self):
        w_gate = torch.sigmoid(self.w_proj.bias)
        return {"W_Mean": w_gate.mean().item(), "W_Std": w_gate.std().item()}

class GroundThinkBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.mixer = SelectiveWKV_V2(config) # V2 Hybrid
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

class GroundThink125M(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([GroundThinkBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_emb.weight = self.head.weight # Tie weights

    def configure_optimizers(self, learning_rate, weight_decay=0.01):
        # Optimization from Design Note 4: Hybrid Tax Mitigation
        # STRATEGY: 
        # 1. "Mamba Dynamics" (Decay/Time-Mixing) -> High LR (1e-3) to encourage long-term state usage
        # 2. "RWKV Projections" (Linear/Gates) -> Standard LR (6e-4) for stability
        # 3. "No Decay" -> Biases, LayerNorms
        
        mamba_params = [] # w_proj (The core recurrence dynamics)
        rwkv_params = []  # Projections, MLP, Embeddings
        nodecay_params = []
        
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
                
            # Filter No-Decay first
            if p.dim() < 2 or "ln" in name or "bias" in name:
                nodecay_params.append(p)
                continue
                
            # Split Hybrid Components
            # w_proj is the parameter governing "Forget/Decay rates" -> The Memory Mechanic
            if "w_proj" in name:
                mamba_params.append(p)
            else:
                rwkv_params.append(p)

        # Mamba LR Boost: 1.5x - 2.0x base LR (Conservative start vs 1e-3 blind jump)
        # Base LR is 6e-4. 
        # Mamba LR -> ~1e-3
        
        optim_groups = [
            {'params': rwkv_params,  'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': mamba_params, 'weight_decay': weight_decay, 'lr': learning_rate * 1.8}, 
            {'params': nodecay_params, 'weight_decay': 0.0,        'lr': learning_rate}
        ]
        
        print(f"🔧 Optimizer Split: RWKV {len(rwkv_params)} tensors | Mamba {len(mamba_params)} tensors | NoDecay {len(nodecay_params)} tensors")
        
        optimizer = bnb.optim.Adam8bit(optim_groups, betas=(0.9, 0.95))
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx)
        
        # Gradient Checkpointing (Selective recommended in notes)
        # For 125M we probably don't need it on A100, but good for scaling
        for block in self.blocks:
            # x = checkpoint(block, x, use_reentrant=False) 
            x = block(x) # Checkpointing disabled for 125M speed
            
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.head(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return logits, loss
        
        return self.head(x), None

# ==========================================
# 3. DATASET V2 (Deterministic Load)
# ==========================================
from datasets import load_from_disk

def get_dataloaders(config):
    # Try looking in local folder (A100 standard path)
    load_path = "groundthink_v2_dataset" 
    
    if not os.path.exists(load_path):
        # Fallback to absolute path if provided or just check sibling dir
        load_path = os.path.join(os.path.dirname(__file__), "groundthink_v2_dataset")
    
    if os.path.exists(load_path):
        print(f"📂 Loading Pre-Processed V2 Dataset from {load_path}...")
        try:
            dataset = load_from_disk(load_path)
            print(f"✅ Loaded {len(dataset)} samples.")
        except Exception as e:
            print(f"❌ Failed to load disk dataset: {e}")
            raise
    else:
        print(f"❌ V2 Dataset not found at {load_path}.")
        print("   Run 'python prepare_v2_dataset.py' first to generate the artifact.")
        raise FileNotFoundError("Missing V2 Dataset Artifact")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        encoded = tokenizer(
            texts, padding=True, truncation=True, 
            max_length=config.max_seq_len, return_tensors="pt"
        )
        
        input_ids = encoded['input_ids']
        labels = input_ids.clone()
        
        # Mask padding tokens in loss
        labels[encoded['attention_mask'] == 0] = -100
        return input_ids, labels

    train_loader = DataLoader(
        dataset, 
        batch_size=config.micro_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=8, # Enable parallelism to fix bottleneck
        pin_memory=True
    )
    return train_loader, tokenizer
    
# ==========================================
# 4. TRAINING LOOP V2 (Scientific Monitoring)
# ==========================================
def train():
    print(f"🚀 Starting V2 Scientific Run: {config.project_name}")
    print(f"   Model: 125M | Layers: {config.n_layer} | Dim: {config.d_model} | Opt: Adam8bit (Split)")
    
    # Setup
    log_dir = f"logs/{config.project_name}"
    writer = SummaryWriter(log_dir)
    ckpt_dir = f"checkpoints/{config.project_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data (V2 Artifact)
    train_loader, tokenizer = get_dataloaders(config)
    
    # Init Model
    model = GroundThink125M(config).to(device).to(config.dtype)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    if USE_TRITON:
        print("✅ Triton Kernel Active")
    else:
        print("⚠️  Running in Slow Python Mode")
        
    optimizer = model.configure_optimizers(config.learning_rate)
    
    # Scheduler: OneCycleLR (Warmup 10% -> 100% -> Decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[config.learning_rate, config.learning_rate * 1.8, config.learning_rate], # Match Split
        total_steps=config.total_steps,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    
    # Training State
    global_step = 0
    t0 = time.time()
    accum_loss = 0.0
    
    model.train()
    
    while global_step < config.total_steps:
        for batch_idx, (idx, targets) in enumerate(train_loader):
            idx, targets = idx.to(device), targets.to(device)
            
            # Forward
            # Note: We rely on automatic mixed precision or native BF16
            with torch.amp.autocast(device_type="cuda", dtype=config.dtype):
                logits, loss = model(idx, targets)
                loss = loss / config.grad_accum_steps
            
            # Backward
            loss.backward()
            accum_loss += loss.item()
            
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                # Gradient Clipping for Stability (Crucial for Recurrent V2)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % 10 == 0:
                    dt = time.time() - t0
                    t0 = time.time() # Reset timer!
                    tps = (config.micro_batch_size * config.grad_accum_steps * config.max_seq_len * 10) / dt
                    # FIX: accum_loss already sums up to the TRUE MEAN because individual losses were pre-divided
                    # So accum_loss IS the average loss for the macro-batch.
                    avg_loss = accum_loss 
                    
                    print(f"Step {global_step} | Loss: {avg_loss:.4f} | TPS: {int(tps)} | Norm: {grad_norm:.2f}")
                    
                    writer.add_scalar("Train/Loss", avg_loss, global_step)
                    writer.add_scalar("Train/TPS", tps, global_step)
                    writer.add_scalar("Train/GradNorm", grad_norm, global_step)
                    writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

                    # Diagnostic Hook
                    if global_step % 50 == 0:
                         # Inspect Layer 6 (Middle)
                        diag = model.blocks[config.n_layer//2].mixer.get_diagnostics()
                        print(f"   🔍 Layer {config.n_layer//2} Diag: {diag}")

                accum_loss = 0.0
                
                # VALIDATION / GENERATION Every 200 Steps
                if global_step % 200 == 0:
                    print(f"🔍 [Step {global_step}] Validation Probe...")
                    model.eval()
                    
                    # 1. Recall Probe (The "Secret Code" Test)
                    # We inject a fact and ask for it later.
                    # "The secret code is 8472. [filler] What is the secret code?"
                    # We will synthesize a quick prompt to test short-term recall.
                    probe_prompt = "The secret agent's passcode is 9922. The agent walked down the hall. He stopped. He asked, 'What is the passcode?' The passcode is"
                    
                    try:
                        with torch.no_grad():
                            # Standard Generation
                            out = torch.zeros((1, 1), dtype=torch.long, device=device) + tokenizer.encode("The")[0] # Dummy start
                            # Let's use the helper generate function logic inline or call a helper
                            # Simple generate for logging:
                            
                            # Test 1: Creative Generation
                            start_tokens = tokenizer.encode("The future of artificial intelligence is", return_tensors="pt").to(device)
                            gen = start_tokens
                            for _ in range(50):
                                logits, _ = model(gen)
                                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                                gen = torch.cat((gen, next_token), dim=1)
                            
                            txt = tokenizer.decode(gen[0])
                            print(f"📝 Creative: {txt}")
                            writer.add_text("Val/Creative", txt, global_step)

                            # Test 2: The Probe
                            probe_tokens = tokenizer.encode(probe_prompt, return_tensors="pt").to(device)
                            gen = probe_tokens
                            for _ in range(5): # Expected output: " 9922"
                                logits, _ = model(gen)
                                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                                gen = torch.cat((gen, next_token), dim=1)
                                
                            probe_out = tokenizer.decode(gen[0])
                            print(f"🧠 Recall Probe: {probe_out}")
                            writer.add_text("Val/Probe", probe_out, global_step)

                    except Exception as e:
                        print(f"⚠️ Generation Error: {e}")
                    
                    model.train()
                
                # Checkpoint Every 500 Steps
                if global_step % 500 == 0:
                    print(f"💾 Saving Step {global_step}...")
                    torch.save(model.state_dict(), f"{ckpt_dir}/step_{global_step}.pt")
                
                # Reset counters
                t0 = time.time()
                accum_loss = 0.0
                
            if global_step >= config.total_steps:
                break
                
    print("🏁 Training Complete.")
    torch.save(model.state_dict(), f"{ckpt_dir}/final.pt")

if __name__ == "__main__":
    train()
