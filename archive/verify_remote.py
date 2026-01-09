import torch
import torch.nn as nn
from transformers import AutoTokenizer
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    vocab_size = 50257
    d_model = 2048
    n_layer = 24
    head_size = 64
    max_seq_len = 2048
    dtype = torch.float16 # FP16 for Inference

# ==========================================
# MODEL DEFINITION (Zero-Dependency)
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
        
        w = torch.sigmoid(self.w_proj(self.x_proj(x))) 
        k = self.k_proj(x)
        v = self.v_proj(x)
        r = torch.sigmoid(self.r_proj(x))
        
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)
        w = w.view(B, T, self.n_head, self.head_size)
        r = r.view(B, T, self.n_head, self.head_size)
        
        # SLOW PATH (No Triton Needed)
        # Sufficient for short validation generation
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
            x = block(x)
        x = self.ln_f(x)
        return self.head(x), None

# ==========================================
# GENERATION LOGIC
# ==========================================
def generate(model, tokenizer, prompt, max_new_tokens=50):
    print(f"\n🧠 Prompt: '{prompt}'")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        out = input_ids
        for _ in range(max_new_tokens):
            logits, _ = model(out)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat((out, next_token), dim=1)
            
    decoded = tokenizer.decode(out[0])
    print(f"🤖 Output: {decoded}")
    return decoded

def main():
    print("🔍 Looking for checkpoints...")
    # Search logic: Look in standard path
    search_path = "checkpoints/groundthink_1B_A100/step_*.pt"
    checkpoints = glob.glob(search_path)
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {search_path}")
        print(f"Current Dir: {os.getcwd()}")
        print(f"Contents: {os.listdir('.')}")
        return

    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_ckpt = checkpoints[-1]
    
    print(f"📂 Loading latest checkpoint: {latest_ckpt}")
    
    config = Config()
    model = GroundThink1B(config)
    model.load_state_dict(torch.load(latest_ckpt, map_location='cuda'))
    model.to('cuda', dtype=torch.float16) # FP16 for speed
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("✅ Model loaded. Testing sanity...")
    generate(model, tokenizer, "The future of AI is")
    generate(model, tokenizer, "Once upon a time")
    generate(model, tokenizer, "The scientific method requires")

if __name__ == "__main__":
    main()
