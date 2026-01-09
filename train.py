"""
Train GroundThink - Config-based training

Usage:
  python train.py --config 5M_deep
  python train.py --config 8M_wide
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
from pathlib import Path
import math
import json
import importlib

from layers_v020 import GroundThinkBlock, RMSNorm, get_layer_config, VERSION


class GroundThinkModel(nn.Module):
    def __init__(self, vocab_size=256, dim=256, n_layers=6, n_heads=8, head_dim=32):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            GroundThinkBlock(dim=dim, n_heads=n_heads, head_dim=head_dim)
            for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for block in self.blocks:
            tm = block.time_mixing
            nn.init.uniform_(tm.time_decay.weight, -1.0, -0.5)
            nn.init.normal_(tm.key.weight, std=0.02)
            nn.init.normal_(tm.value.weight, std=0.02)
            nn.init.normal_(tm.receptance.weight, std=0.02)
            nn.init.normal_(tm.gate.weight, std=0.01)
            nn.init.normal_(tm.out_proj.weight, std=0.02 / math.sqrt(2 * len(self.blocks)))
            if tm.use_grounding:
                nn.init.constant_(tm.grounding.base_decay, 0.5)
                nn.init.constant_(tm.grounding.res_weight, 1.0)
    
    def get_param_groups(self, lr_base=3e-4, lr_decay=1e-3, wd=0.1):
        decay_params, nodecay_params, regular_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'time_decay' in name or 'base_decay' in name or 'grounding' in name:
                decay_params.append(param)
            elif param.dim() < 2 or 'ln' in name or 'norm' in name or 'embed' in name:
                nodecay_params.append(param)
            else:
                regular_params.append(param)
        return [
            {'params': decay_params, 'lr': lr_decay, 'weight_decay': 0.0, 'name': 'decay'},
            {'params': nodecay_params, 'lr': lr_base, 'weight_decay': 0.0, 'name': 'nodecay'},
            {'params': regular_params, 'lr': lr_base, 'weight_decay': wd, 'name': 'regular'},
        ]
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.ln_out(x)
        return x @ self.embed.weight.T


class LineDataset(Dataset):
    def __init__(self, filepath, tokenizer, seq_len=256):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        with open(filepath, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if len(line.strip()) > 50]
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.lines[idx])
        if len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        else:
            tokens = tokens[:self.seq_len + 1]
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        for c in ' \n\t.,!?\'"-:;()':
            if c not in chars:
                chars.append(c)
        self.char_to_id = {c: i for i, c in enumerate(sorted(set(chars)))}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


def compute_state_norms(model, sample_input):
    model.eval()
    with torch.no_grad():
        x = model.embed(sample_input)
        norms = []
        for block in model.blocks:
            x, state = block(x)
            if state is not None:
                norms.append(state.norm().item())
    model.train()
    return norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config name: 5M_deep, 8M_wide')
    parser.add_argument('--steps', type=int, default=None, help='Override num_steps')
    args = parser.parse_args()
    
    # Load config
    config_module = importlib.import_module(f'configs.model_{args.config}')
    model_cfg = config_module.MODEL_CONFIG
    train_cfg = config_module.TRAIN_CONFIG.copy()
    save_name = config_module.SAVE_NAME
    
    if args.steps:
        train_cfg['num_steps'] = args.steps
    
    print(f"GroundThink v{VERSION} - {args.config}")
    print(f"Model: {json.dumps(model_cfg, indent=2)}")
    print(f"Train: {json.dumps(train_cfg, indent=2)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    data_path = Path('../data/final_training_mix.txt')
    if not data_path.exists():
        print("ERROR: No training data found!")
        return
    
    print("Building tokenizer...")
    with open(data_path, 'r', encoding='utf-8') as f:
        sample_text = f.read(1_000_000)
    tokenizer = CharTokenizer(sample_text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    dataset = LineDataset(data_path, tokenizer, train_cfg['seq_len'])
    print(f"Dataset: {len(dataset):,} samples")
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=0)
    
    model = GroundThinkModel(
        tokenizer.vocab_size, 
        model_cfg['dim'], 
        model_cfg['n_layers'], 
        model_cfg['n_heads'], 
        model_cfg['head_dim']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    param_groups = model.get_param_groups(train_cfg['lr_base'], train_cfg['lr_decay'], wd=0.1)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    
    for pg in param_groups:
        n = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: {n:,} params, lr={pg['lr']:.1e}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def lr_lambda(step):
        if step < train_cfg['warmup']:
            return step / train_cfg['warmup']
        return 0.5 * (1 + math.cos(math.pi * (step - train_cfg['warmup']) / 
                                    (train_cfg['num_steps'] - train_cfg['warmup'])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    sample_batch = next(iter(loader))[0][:1].to(device)
    
    # Initial state norms
    init_norms = compute_state_norms(model, sample_batch)
    print(f"Initial state norms: {[f'{n:.2f}' for n in init_norms]}")
    
    print(f"\nTraining for {train_cfg['num_steps']} steps (warmup: {train_cfg['warmup']})...")
    model.train()
    step, total_loss, best_loss = 0, 0, float('inf')
    start = time.time()
    
    while step < train_cfg['num_steps']:
        for inputs, targets in loader:
            if step >= train_cfg['num_steps']:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % 100 == 0:
                avg = total_loss / 100
                elapsed = time.time() - start
                tok_s = step * train_cfg['batch_size'] * train_cfg['seq_len'] / elapsed
                current_lr = scheduler.get_last_lr()[0]
                
                # State health check every 500 steps
                state_info = ""
                if step % 500 == 0:
                    norms = compute_state_norms(model, sample_batch)
                    state_info = f" | states: {[f'{n:.1f}' for n in norms[:3]]}"
                
                print(f"Step {step:5d} | Loss: {avg:.4f} | LR: {current_lr:.2e} | "
                      f"Grad: {grad_norm:.2f} | {tok_s:.0f} tok/s{state_info}")
                total_loss = 0
                if avg < best_loss:
                    best_loss = avg
    
    elapsed = time.time() - start
    print(f"\nDone! {elapsed:.1f}s, {step * train_cfg['batch_size'] * train_cfg['seq_len'] / elapsed:.0f} tok/s")
    print(f"Best loss: {best_loss:.4f}")
    
    # Final state norms
    final_norms = compute_state_norms(model, sample_batch)
    print(f"Final state norms: {[f'{n:.2f}' for n in final_norms]}")
    
    # Generate samples
    print("\n--- Narrative Test ---")
    model.eval()
    narrative_prompts = [
        "Once upon a time",
        "The old man looked",
        "She walked into",
    ]
    
    for prompt_text in narrative_prompts:
        print(f"\nPrompt: '{prompt_text}'")
        with torch.no_grad():
            prompt = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
            min_len = 16
            if prompt.shape[1] < min_len:
                pad = torch.zeros(1, min_len - prompt.shape[1], dtype=torch.long, device=device)
                prompt_padded = torch.cat([pad, prompt], dim=1)
            else:
                prompt_padded = prompt
            
            for _ in range(150):
                logits = model(prompt_padded)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)
                prompt_padded = torch.cat([prompt_padded, next_token], dim=1)
            
            generated = tokenizer.decode(prompt_padded[0, min_len - prompt.shape[1]:].tolist())
            if '\n' in generated:
                generated = generated[:generated.index('\n')]
            print(f"  → {generated[:200]}")
    
    print("\n--- Conversation Test ---")
    prompts = ["What is your name?", "How are you today?", "Can you help me?"]
    
    for prompt_text in prompts:
        print(f"\nQ: {prompt_text}")
        with torch.no_grad():
            prompt = torch.tensor([tokenizer.encode(prompt_text)]).to(device)
            min_len = 16
            if prompt.shape[1] < min_len:
                pad = torch.zeros(1, min_len - prompt.shape[1], dtype=torch.long, device=device)
                prompt_padded = torch.cat([pad, prompt], dim=1)
            else:
                prompt_padded = prompt
            
            for _ in range(100):
                logits = model(prompt_padded)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)
                prompt_padded = torch.cat([prompt_padded, next_token], dim=1)
            
            generated = tokenizer.decode(prompt_padded[0, min_len - prompt.shape[1]:].tolist())
            if '\n' in generated:
                generated = generated[:generated.index('\n')]
            print(f"A: {generated[:150]}")
    
    # Save
    save_path = f'{save_name}_{train_cfg["num_steps"]//1000}k.pt'
    torch.save({
        'model': model.state_dict(),
        'vocab': tokenizer.char_to_id,
        'model_config': model_cfg,
        'train_config': train_cfg,
        'version': VERSION,
        'best_loss': best_loss,
    }, save_path)
    print(f"\n✅ Saved to {save_path}")


if __name__ == '__main__':
    main()
