import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from transformers import AutoTokenizer

USE_TRITON = False # Force slow kernel for compatibility on local infer

class Config:
    vocab_size = 50257      # GPT2 Standard
    d_model = 2048          # 1B Scale
    n_layer = 24            # Deeper than T4 version (18 -> 24)
    head_size = 64          
    max_seq_len = 2048
    project_name = "groundthink_1B_A100"
    dtype = torch.float16   # Local infer

config = Config()

# ==========================================
# MODEL DEFINITION (Copied from training script)
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
        
        k = k.view(B, T, self.n_head, self.head_size) 
        v = v.view(B, T, self.n_head, self.head_size) 
        w = w.view(B, T, self.n_head, self.head_size) 
        r = r.view(B, T, self.n_head, self.head_size)
    
        # SLOW PATH for Inference Compatibility
        k = k.unsqueeze(-1); v = v.unsqueeze(-2); w = w.unsqueeze(-1); r = r.unsqueeze(-2)
        if state is None: 
            state = torch.zeros(B, self.n_head, self.head_size, self.head_size, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            kv = k[:, t] @ v[:, t]
            state = (1 - w[:, t]) * state + kv
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
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x) # Checkpointing not needed for inference
        x = self.ln_f(x)
        return self.head(x), None

# ==========================================
# INFERENCE LOGIC
# ==========================================
def generate(model, tokenizer, prompt, max_new_tokens=20, temperature=1.0, top_k=50, repetition_penalty=1.2):
    model.eval()
    idx = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    
    print(f"\nExample Prompt: {prompt}")
    print("Generating (with debug info)...")
    
    # Keep track of generated tokens for repetition penalty
    generated = idx.clone()

    with torch.no_grad():
        for i in range(max_new_tokens):
            logits, _ = model(generated)
            logits = logits[:, -1, :] 
            
            # 1. Apply Repetition Penalty
            # Penalize tokens that have already been generated
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty

            # 2. Temperature
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # 3. Top-K Inspection (Debug)
            if i < 3: # Only show first few steps
                top_probs, top_indices = torch.topk(probs, 5)
                print(f"\nStep {i+1} Top candidates:")
                for p, idx_val in zip(top_probs[0], top_indices[0]):
                    token_str = tokenizer.decode([idx_val.item()])
                    print(f"  '{token_str}' ({p.item():.4f})")
            
            # 4. Sampling
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat((generated, next_token), dim=1)
            print(tokenizer.decode(next_token[0]), end="", flush=True)

    print("\n\nDone.")

def main():
    print("🧠 Loading GroundThink 1B Inference...")
    
    # 1. Setup Model
    model = GroundThink1B(config)
    
    # 2. Load Checkpoint
    checkpoint_path = r"models\step_500.pt" 
    
    try:
        if not os.path.exists(checkpoint_path):
             # Fallback if user downloaded to root
             if os.path.exists("step_500.pt"):
                checkpoint_path = "step_500.pt"
             else:
                raise FileNotFoundError(f"Could not find model at {checkpoint_path} or step_500.pt")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cuda')
        
        # Handle cleanup if needed
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        model.load_state_dict(state_dict)
        model.to('cuda', dtype=torch.float16) # FP16 fits in 6GB easier than BF16 if driver issue
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 4. Generate
    generate(model, tokenizer, "Once upon a time, there was a little robot who")
    generate(model, tokenizer, "The king said to the knight, 'Go and find")

if __name__ == "__main__":
    import os
    if torch.cuda.is_available():
        main()
    else:
        print("❌ CUDA required for this inference script.")
