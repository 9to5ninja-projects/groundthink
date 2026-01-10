#!/usr/bin/env python3
"""
Long-Range Dependency (LRD) Test for Character-Level Models

Tests whether the model benefits from longer context. This is the appropriate
test for char-level models (unlike NIAH which requires retrieval capability).

Key insight: Char-level models predict based on language patterns, not retrieval.
A successful LRD test shows that perplexity improves with more context.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model
from data.tokenizer import CharTokenizer


def load_shakespeare():
    """Load Shakespeare text."""
    data_path = Path(__file__).parent.parent / "data" / "shakespeare.txt"
    with open(data_path, 'r', encoding='utf-8') as f:
        return f.read()


def test_lrd_single(
    model,
    tokenizer,
    text: str,
    context_lengths: list[int] = [8, 16, 32, 64, 128],
    device: str = 'cuda',
) -> dict:
    """
    Test if model benefits from longer context on a single text sample.
    
    Returns dict with loss/ppl at each context length.
    """
    tokens = tokenizer.encode(text)
    full_len = len(tokens)
    
    results = {}
    for ctx_len in context_lengths:
        if ctx_len >= full_len - 1:
            continue
        
        # Use last ctx_len tokens, predict each except first
        start = full_len - ctx_len
        x = torch.tensor([tokens[start:-1]], device=device)
        y = torch.tensor([tokens[start+1:]], device=device)
        
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            ppl = torch.exp(loss).item()
            results[ctx_len] = {'loss': loss.item(), 'ppl': ppl}
    
    return results


def run_lrd_test(
    model_name: str,
    context_lengths: list[int] = [8, 16, 32, 64, 128, 256],
    num_samples: int = 10,
    sample_size: int = 300,  # chars per sample
    device: str = 'cuda',
    checkpoint_path: str = None,
) -> dict:
    """
    Run Long-Range Dependency test on a model.
    
    Tests whether the model benefits from longer context by measuring
    perplexity improvement as context increases.
    """
    print(f"\n{'='*60}")
    print(f"Long-Range Dependency (LRD) Test: {model_name}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load model
    model = get_model(model_name).to(device)
    
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint (step {ckpt.get('step', '?')})")
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Load text and tokenizer
    shakespeare = load_shakespeare()
    tokenizer = CharTokenizer(text=shakespeare)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()
    
    # Aggregate results across samples
    aggregated = {ctx: {'losses': [], 'ppls': []} for ctx in context_lengths}
    
    # Sample from different parts of text
    text_len = len(shakespeare)
    sample_starts = [i * (text_len // num_samples) for i in range(num_samples)]
    
    for i, start in enumerate(sample_starts):
        sample = shakespeare[start:start + sample_size]
        if len(sample) < sample_size:
            continue
        
        results = test_lrd_single(model, tokenizer, sample, context_lengths, device)
        
        for ctx_len, metrics in results.items():
            aggregated[ctx_len]['losses'].append(metrics['loss'])
            aggregated[ctx_len]['ppls'].append(metrics['ppl'])
    
    # Compute averages
    print("Results (averaged over samples):")
    print("-" * 45)
    
    summary = {}
    valid_ctx_lens = []
    for ctx_len in sorted(context_lengths):
        if aggregated[ctx_len]['losses']:
            avg_loss = sum(aggregated[ctx_len]['losses']) / len(aggregated[ctx_len]['losses'])
            avg_ppl = sum(aggregated[ctx_len]['ppls']) / len(aggregated[ctx_len]['ppls'])
            summary[ctx_len] = {'loss': avg_loss, 'ppl': avg_ppl}
            valid_ctx_lens.append(ctx_len)
            print(f"  ctx={ctx_len:3d}: loss={avg_loss:.4f}  PPL={avg_ppl:.2f}")
    
    # Calculate improvement from shortest to longest context
    if len(valid_ctx_lens) >= 2:
        shortest = min(valid_ctx_lens)
        longest = max(valid_ctx_lens)
        loss_short = summary[shortest]['loss']
        loss_long = summary[longest]['loss']
        improvement = (loss_short - loss_long) / loss_short * 100
        
        print()
        print(f"Context sensitivity: {improvement:+.1f}% loss reduction ({shortest}→{longest})")
        
        if improvement > 15:
            print("✅ STRONG: Model significantly benefits from longer context")
            verdict = 'strong'
        elif improvement > 5:
            print("✅ GOOD: Model uses longer context effectively")
            verdict = 'good'
        elif improvement > 0:
            print("⚠️ WEAK: Model slightly benefits from context")
            verdict = 'weak'
        else:
            print("❌ FAIL: Model does not benefit from longer context")
            verdict = 'fail'
    else:
        improvement = 0
        verdict = 'insufficient_data'
    
    return {
        'model': model_name,
        'summary': summary,
        'improvement': improvement,
        'verdict': verdict,
    }


def compare_models(
    model_names: list[str],
    checkpoint_paths: dict[str, str] = None,
    device: str = 'cuda',
) -> None:
    """Compare multiple models on LRD test."""
    
    print("\n" + "="*70)
    print("LRD MODEL COMPARISON")
    print("="*70)
    
    results = {}
    checkpoint_paths = checkpoint_paths or {}
    
    for name in model_names:
        try:
            ckpt = checkpoint_paths.get(name)
            result = run_lrd_test(name, checkpoint_path=ckpt, device=device)
            results[name] = result
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            results[name] = {'error': str(e)}
    
    # Summary table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Improvement':>12} {'Verdict':>10}")
    print("-"*40)
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name:<15} {'ERROR':>12} {'N/A':>10}")
        else:
            imp = result.get('improvement', 0)
            verdict = result.get('verdict', '?')
            print(f"{name:<15} {imp:>+11.1f}% {verdict:>10}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Long-Range Dependency Test')
    parser.add_argument('--model', '-m', type=str, default='HY',
                       help='Model variant to test')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                       help='Path to checkpoint')
    parser.add_argument('--context', '-ctx', nargs='+', type=int,
                       default=[8, 16, 32, 64, 128],
                       help='Context lengths to test')
    parser.add_argument('--samples', '-n', type=int, default=10,
                       help='Number of text samples')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare common variants
        compare_models(['HY', 'GF', 'GF-MH'], device='cuda')
    else:
        run_lrd_test(
            args.model,
            context_lengths=args.context,
            num_samples=args.samples,
            checkpoint_path=args.checkpoint,
        )
