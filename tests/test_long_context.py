"""
Long-Context Degradation Test (Task 60)

Measure perplexity at 64, 128, 256, 512 tokens.
Check for smooth degradation vs catastrophic failure.

Usage:
    python tests/test_long_context.py

Reference: VALIDATION_ROADMAP.md D4
Created: 2026-01-11
"""

import os
os.environ['CXX'] = '/usr/bin/g++'
os.environ['CC'] = '/usr/bin/gcc'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from models import get_model
from tools.thresholds import check_status

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 16000


def long_context_test(model_name: str = 'GF-MH'):
    """Test perplexity degradation over sequence lengths."""
    print(f"\n{'='*50}")
    print(f" Long-Context Degradation Test (Task 60)")
    print(f"{'='*50}")
    print(f" Model: {model_name} | Device: {DEVICE}")
    
    model = get_model(model_name, vocab_size=VOCAB_SIZE).to(DEVICE)
    model.eval()
    
    lengths = [64, 128, 256, 512]
    results = {}
    
    with torch.no_grad():
        for seq_len in lengths:
            x = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=DEVICE)
            logits = model(x)
            
            # Compute loss on last 32 tokens
            eval_len = min(32, seq_len - 1)
            logits_tail = logits[:, -eval_len-1:-1, :]
            targets_tail = x[:, -eval_len:]
            
            loss = F.cross_entropy(
                logits_tail.reshape(-1, logits_tail.size(-1)),
                targets_tail.reshape(-1)
            ).item()
            
            ppl = 2.718 ** loss  # Approximate perplexity
            results[seq_len] = {'loss': loss, 'ppl': ppl}
            print(f"   seq_len={seq_len:3d}: loss={loss:.3f} ppl={ppl:.1f}")
    
    # Compute degradation ratio (512 vs 64)
    ppl_64 = results[64]['ppl']
    ppl_512 = results[512]['ppl']
    ratio = ppl_512 / ppl_64
    
    status = check_status('lc_degradation_64_512', ratio)
    
    print(f"\n PPL ratio (512/64): {ratio:.2f}x")
    print(f" Status: {status}")
    print(f"{'='*50}\n")
    
    return {
        'results': results,
        'ratio': ratio,
        'status': status,
    }


if __name__ == '__main__':
    long_context_test()
