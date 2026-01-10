#!/usr/bin/env python3
"""
Char-Level Needle-in-a-Haystack (NIAH) Test

Adapted NIAH for character-level models trained on Shakespeare.
Tests retrieval ability across context positions without requiring
word-level understanding.

How it works:
1. Create "needle" = unique pattern (e.g., "XYZZY42")
2. Create "haystack" = Shakespeare text (filler)
3. Insert needle at various positions (0%, 25%, 50%, 75%, 100%)
4. Ask model to complete the pattern after seeing full context
5. Measure if model can retrieve/predict the needle pattern

This tests the CORE HYPOTHESIS: Can the hybrid architecture
retrieve information from different positions in long context?
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model
from data.tokenizer import CharTokenizer


def load_shakespeare():
    """Load Shakespeare text for haystack filler."""
    data_path = Path(__file__).parent.parent / "data" / "shakespeare.txt"
    with open(data_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_tokenizer(text: str) -> CharTokenizer:
    """Create proper CharTokenizer from text."""
    return CharTokenizer(text=text)


def encode(text: str, tokenizer: CharTokenizer) -> list[int]:
    """Encode text to token indices using tokenizer."""
    return tokenizer.encode(text)


def decode(indices: list[int], tokenizer: CharTokenizer) -> str:
    """Decode token indices to text using tokenizer."""
    return tokenizer.decode(indices)


def generate_needle(seed: int = 42, vocab_chars: str = None) -> str:
    """
    Generate a unique pattern that's rare in Shakespeare but uses valid chars.
    Uses uppercase letters (rare but valid in corpus).
    Format: XYZQW (5 uppercase letters - very rare pattern in Shakespeare)
    """
    import random
    random.seed(seed)
    
    # Use only chars that exist in vocab
    if vocab_chars:
        # Filter to uppercase letters that are in vocab
        valid_upper = [c for c in 'XYZQWKJV' if c in vocab_chars]
        if len(valid_upper) < 3:
            valid_upper = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c in vocab_chars]
    else:
        valid_upper = list('XYZQWKJV')
    
    # Create rare pattern: 5 uppercase in sequence (very rare in Shakespeare)
    pattern = ''.join(random.choices(valid_upper, k=7))
    return pattern


def create_haystack_with_needle(
    shakespeare: str,
    needle: str,
    context_length: int,
    depth_percent: float,  # 0.0 = start, 1.0 = end
) -> str:
    """
    Create haystack text with needle inserted at specified depth.
    
    Args:
        shakespeare: Source text for filler
        needle: Pattern to insert
        context_length: Total length of haystack
        depth_percent: Where to insert (0.0=start, 0.5=middle, 1.0=end)
    """
    # Calculate filler needed
    filler_length = context_length - len(needle)
    
    # Get random chunk of Shakespeare
    import random
    max_start = len(shakespeare) - filler_length - 100
    start_idx = random.randint(0, max(0, max_start))
    filler = shakespeare[start_idx:start_idx + filler_length]
    
    # Insert needle at depth
    insert_pos = int(len(filler) * depth_percent)
    haystack = filler[:insert_pos] + needle + filler[insert_pos:]
    
    return haystack[:context_length]


def test_retrieval(
    model,
    haystack: str,
    needle: str,
    tokenizer: CharTokenizer,
    device: str = 'cuda',
) -> dict:
    """
    Test if model can retrieve the needle pattern.
    
    Strategy: 
    1. Feed haystack as context
    2. Generate next tokens
    3. Check if generated tokens match needle pattern
    """
    model.eval()
    
    # Find needle position in haystack
    needle_pos = haystack.find(needle)
    if needle_pos == -1:
        return {'success': False, 'error': 'needle not in haystack'}
    
    # Create prompt: everything up to and including first char of needle
    prompt = haystack[:needle_pos + 1]  # Include '[' of needle
    target = needle[1:]  # Rest of needle to predict
    
    # Encode using proper tokenizer
    prompt_ids = encode(prompt, tokenizer)
    # Pad to multiple of 8 for Mamba stride requirements
    pad_len = (8 - len(prompt_ids) % 8) % 8
    prompt_ids = [0] * pad_len + prompt_ids
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    
    # Generate
    with torch.no_grad():
        generated = []
        current = prompt_tensor
        
        for _ in range(len(target)):
            logits = model(current)
            next_logit = logits[0, -1, :]  # Last position
            next_token = next_logit.argmax().item()
            generated.append(next_token)
            # Append and keep padded to 8
            current = torch.cat([
                current,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
    
    generated_text = decode(generated, tokenizer)
    
    # Check match
    exact_match = generated_text == target
    partial_match = sum(g == t for g, t in zip(generated_text, target)) / len(target)
    
    return {
        'success': exact_match,
        'partial_match': partial_match,
        'expected': target,
        'generated': generated_text,
        'needle_pos': needle_pos,
        'context_length': len(haystack),
    }


def run_niah_test(
    model_name: str,
    context_lengths: list[int] = [256, 512, 1024, 2048],
    depth_percents: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    num_trials: int = 3,
    device: str = 'cuda',
    checkpoint_path: str = None,
) -> dict:
    """
    Run full NIAH test suite on a model.
    
    Args:
        model_name: Model variant to test (e.g., 'tiny', 'GF-MH')
        context_lengths: List of context sizes to test
        depth_percents: Where in context to place needle
        num_trials: Repetitions per condition
        device: 'cuda' or 'cpu'
        checkpoint_path: Optional path to trained checkpoint
    """
    print(f"\n{'='*60}")
    print(f"NIAH Test: {model_name}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load model
    model = get_model(model_name).to(device)
    
    # Load checkpoint if provided
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
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Load Shakespeare and build tokenizer
    shakespeare = load_shakespeare()
    tokenizer = create_tokenizer(shakespeare)
    vocab_chars = ''.join(tokenizer.char_to_id.keys())
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Results storage
    results = {
        'model': model_name,
        'trials': [],
        'summary': {},
    }
    
    # Run tests
    for ctx_len in context_lengths:
        for depth in depth_percents:
            successes = 0
            partial_scores = []
            
            for trial in range(num_trials):
                # Different needle each trial (using valid vocab chars)
                needle = generate_needle(
                    seed=trial * 100 + int(depth * 100) + ctx_len,
                    vocab_chars=vocab_chars
                )
                
                # Create haystack
                haystack = create_haystack_with_needle(
                    shakespeare, needle, ctx_len, depth
                )
                
                # Test
                result = test_retrieval(
                    model, haystack, needle,
                    tokenizer, device
                )
                
                successes += result['success']
                partial_scores.append(result['partial_match'])
                
                results['trials'].append({
                    'context_length': ctx_len,
                    'depth': depth,
                    'trial': trial,
                    **result
                })
            
            # Summary for this condition
            accuracy = successes / num_trials
            avg_partial = sum(partial_scores) / len(partial_scores)
            
            depth_label = {0.0: 'start', 0.25: 'early', 0.5: 'middle', 
                          0.75: 'late', 1.0: 'end'}[depth]
            
            status = '✅' if accuracy >= 0.5 else '⚠️' if accuracy > 0 else '❌'
            print(f"  ctx={ctx_len:4d} depth={depth_label:6s}: "
                  f"{accuracy*100:5.1f}% exact, {avg_partial*100:5.1f}% partial {status}")
            
            results['summary'][(ctx_len, depth)] = {
                'accuracy': accuracy,
                'partial': avg_partial,
            }
    
    # Overall summary
    all_accuracies = [v['accuracy'] for v in results['summary'].values()]
    overall = sum(all_accuracies) / len(all_accuracies)
    print(f"\n  Overall accuracy: {overall*100:.1f}%")
    results['overall_accuracy'] = overall
    
    return results


def compare_models(
    model_names: list[str],
    context_lengths: list[int] = [256, 512, 1024],
    device: str = 'cuda',
) -> dict:
    """
    Compare multiple models on NIAH test.
    """
    print("\n" + "="*70)
    print("NIAH COMPARISON TEST")
    print("="*70)
    
    all_results = {}
    
    for model_name in model_names:
        try:
            results = run_niah_test(
                model_name,
                context_lengths=context_lengths,
                device=device,
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Overall':>10} {'@256':>8} {'@512':>8} {'@1024':>8}")
    print("-"*50)
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"{model_name:<15} ERROR: {results['error'][:30]}")
            continue
        
        overall = results['overall_accuracy'] * 100
        
        # Get accuracy at each context length (averaged across depths)
        ctx_accs = {}
        for ctx_len in context_lengths:
            accs = [v['accuracy'] for k, v in results['summary'].items() if k[0] == ctx_len]
            ctx_accs[ctx_len] = sum(accs) / len(accs) * 100 if accs else 0
        
        print(f"{model_name:<15} {overall:>9.1f}% "
              f"{ctx_accs.get(256, 0):>7.1f}% "
              f"{ctx_accs.get(512, 0):>7.1f}% "
              f"{ctx_accs.get(1024, 0):>7.1f}%")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Char-level NIAH Test")
    parser.add_argument('--model', '-m', type=str, default='tiny',
                        help='Model to test (default: tiny)')
    parser.add_argument('--checkpoint', '-k', type=str, default=None,
                        help='Path to trained checkpoint (e.g., checkpoints/ckpt_HY_final.pt)')
    parser.add_argument('--compare', '-c', nargs='+', type=str,
                        help='Compare multiple models (e.g., --compare tiny small GF-MH)')
    parser.add_argument('--context', '-x', type=int, nargs='+', 
                        default=[256, 512, 1024],
                        help='Context lengths to test')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare, args.context, args.device)
    else:
        run_niah_test(args.model, args.context, device=args.device, 
                      checkpoint_path=args.checkpoint)
