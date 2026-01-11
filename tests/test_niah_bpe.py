"""
BPE Needle-in-a-Haystack (NIAH) Test for Task 0.0.1
GroundThink — Phase 0: Base Model Characterization
Copyright (c) 2026 Matthew [m_tes]

ATTRIBUTION:
Needle-in-a-Haystack (NIAH) testing is a standard evaluation method.
Concept from information retrieval and context window research.

Our adaptation from test_niah_char.py (character-level version).

OUR CONTRIBUTION:
    - Adaptation for BPE tokenization (subword-level testing)
    - Integration with WikiText-103 corpus
    - Multi-depth testing protocol (0%, 25%, 50%, 75%, 100%)
    - Greedy decoding evaluation methodology

See ATTRIBUTION.md for full citation details.
Adapted for BPE tokenization (WikiText-103 models).
Tests retrieval ability across context positions at word/subword level.

How it works:
1. Create "needle" = unique phrase (e.g., "The unique identifier is: 7423")
2. Create "haystack" = WikiText-103 text (filler)
3. Insert needle at various positions (0%, 25%, 50%, 75%, 100%)
4. Ask model to complete retrieval prompt after seeing full context
5. Measure if model can retrieve information from different positions

Usage:
    python tests/test_niah_bpe.py --checkpoint checkpoints/task_0_0_1/ckpt_001000.pt \\
                                  --context_length 512 --depths 0.0 0.25 0.5 0.75 1.0
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import random

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rwkv6_pure import create_rwkv6_4m
from data import load_stateful_dataset


def load_wikitext_corpus():
    """Load WikiText-103 text for haystack filler."""
    # Use a subset of WikiText-103 for filler
    corpus_path = Path(__file__).parent.parent / "data" / "wikitext-103" / "wiki.train.tokens"
    
    if not corpus_path.exists():
        print(f"Warning: WikiText corpus not found at {corpus_path}")
        print("Using fallback filler text")
        return " ".join(["Sample text. " * 1000])  # Fallback
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


def generate_needle(seed: int = 42) -> str:
    """
    Generate a unique retrieval pattern.
    Format: "The secret code is: XXXX" where XXXX is a 4-digit number.
    """
    random.seed(seed)
    code = random.randint(1000, 9999)
    needle = f"The secret code is: {code}"
    return needle, str(code)


def create_haystack_with_needle(
    corpus: str,
    needle: str,
    tokenizer,
    context_length: int,
    depth_percent: float,  # 0.0 = start, 1.0 = end
) -> tuple[list[int], int]:
    """
    Create haystack tokens with needle inserted at specified depth.
    
    Args:
        corpus: Source text for filler
        needle: Pattern to insert
        tokenizer: BPE tokenizer
        context_length: Total length in tokens
        depth_percent: Where to insert (0.0=start, 0.5=middle, 1.0=end)
    
    Returns:
        tokens: List of token indices
        needle_start_idx: Where needle starts in token sequence
    """
    # Tokenize needle
    needle_tokens = tokenizer.encode(needle)
    needle_len = len(needle_tokens)
    
    # Calculate filler length
    filler_len = context_length - needle_len
    
    # Get random chunk of corpus for filler
    max_start = len(corpus) - filler_len * 10  # Estimate ~10 chars per token
    start_idx = random.randint(0, max(0, max_start))
    filler_text = corpus[start_idx:start_idx + filler_len * 10]
    filler_tokens = tokenizer.encode(filler_text)
    
    # Truncate to exact length
    if len(filler_tokens) > filler_len:
        filler_tokens = filler_tokens[:filler_len]
    elif len(filler_tokens) < filler_len:
        # Pad with repeating filler if needed
        while len(filler_tokens) < filler_len:
            filler_tokens.extend(filler_tokens[:min(100, filler_len - len(filler_tokens))])
        filler_tokens = filler_tokens[:filler_len]
    
    # Insert needle at depth
    needle_start_idx = int(filler_len * depth_percent)
    
    haystack = (
        filler_tokens[:needle_start_idx] +
        needle_tokens +
        filler_tokens[needle_start_idx:]
    )
    
    # Ensure exact length
    haystack = haystack[:context_length]
    
    return haystack, needle_start_idx


def test_retrieval(
    model,
    haystack_tokens: list[int],
    needle: str,
    answer: str,
    tokenizer,
    device,
    depth_percent: float,
) -> dict:
    """
    Test if model can retrieve needle after seeing full context.
    
    Args:
        model: Trained model
        haystack_tokens: Context with needle
        needle: The needle text
        answer: Expected answer (e.g., "7423")
        tokenizer: BPE tokenizer
        device: torch device
        depth_percent: Where needle was inserted
    
    Returns:
        dict: {
            'depth': float,
            'success': bool,
            'predicted': str,
            'expected': str,
        }
    """
    # Create retrieval prompt
    prompt = "What is the secret code? "
    prompt_tokens = tokenizer.encode(prompt)
    
    # Full context = haystack + prompt
    context = haystack_tokens + prompt_tokens
    
    # Convert to tensor
    x = torch.tensor([context], dtype=torch.long, device=device)
    
    # Generate answer (greedy decoding)
    model.eval()
    with torch.no_grad():
        # Generate up to 10 tokens
        generated = []
        for _ in range(10):
            logits = model(x)  # [1, L, V]
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            
            # Stop at space or newline
            if next_token in [tokenizer.encode(' ')[0], tokenizer.encode('\n')[0]]:
                break
            
            # Append for next step
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
        
        # Decode prediction
        predicted = tokenizer.decode(generated).strip()
    
    # Check if answer is in prediction
    success = answer in predicted
    
    return {
        'depth': depth_percent,
        'success': success,
        'predicted': predicted,
        'expected': answer,
    }


def run_niah_test(
    checkpoint_path: str,
    context_length: int = 512,
    depths: list[float] = None,
    num_trials: int = 5,
):
    """
    Run NIAH test at multiple depths.
    
    Args:
        checkpoint_path: Path to model checkpoint
        context_length: Length of context in tokens
        depths: List of depth percentages (e.g., [0.0, 0.25, 0.5, 0.75, 1.0])
        num_trials: Number of trials per depth (different needles)
    """
    if depths is None:
        depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print(f"\n{'='*60}")
    print(f"NIAH Test (BPE) - Checkpoint: {checkpoint_path}")
    print(f"Context length: {context_length} tokens")
    print(f"Depths: {depths}")
    print(f"Trials per depth: {num_trials}")
    print(f"{'='*60}\n")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer (need to initialize dataset to get tokenizer)
    print("Loading tokenizer...")
    _, tokenizer = load_stateful_dataset(
        'data/wikitext-103',
        batch_size=1,
        seq_len=64,
        scale='LARGE',  # Forces BPE
    )
    
    print(f"Creating model (vocab={tokenizer.vocab_size})...")
    model = create_rwkv6_4m(vocab_size=tokenizer.vocab_size).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load corpus
    print("Loading WikiText corpus...")
    corpus = load_wikitext_corpus()
    
    # Run tests
    results = []
    
    for depth in depths:
        print(f"\nTesting depth {depth*100:.0f}%:")
        depth_results = []
        
        for trial in range(num_trials):
            # Generate needle
            needle, answer = generate_needle(seed=42 + trial)
            
            # Create haystack
            haystack_tokens, needle_pos = create_haystack_with_needle(
                corpus, needle, tokenizer, context_length, depth
            )
            
            # Test retrieval
            result = test_retrieval(
                model, haystack_tokens, needle, answer,
                tokenizer, device, depth
            )
            
            depth_results.append(result)
            
            status = "✓" if result['success'] else "✗"
            print(f"  Trial {trial+1}: {status} (predicted: '{result['predicted']}', expected: '{result['expected']}')")
        
        # Aggregate
        success_rate = sum(r['success'] for r in depth_results) / num_trials
        print(f"  Success rate: {success_rate*100:.1f}%")
        
        results.append({
            'depth': depth,
            'success_rate': success_rate,
            'trials': depth_results,
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("NIAH Test Summary:")
    print(f"{'='*60}")
    print(f"{'Depth':<10} {'Success Rate':<15}")
    print(f"{'-'*25}")
    for r in results:
        print(f"{r['depth']*100:>5.0f}%     {r['success_rate']*100:>5.1f}%")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="NIAH test for BPE models")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--context_length', type=int, default=512,
                       help='Context length in tokens')
    parser.add_argument('--depths', type=float, nargs='+',
                       default=[0.0, 0.25, 0.5, 0.75, 1.0],
                       help='Depth percentages to test')
    parser.add_argument('--num_trials', type=int, default=5,
                       help='Number of trials per depth')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Run test
    results = run_niah_test(
        args.checkpoint,
        args.context_length,
        args.depths,
        args.num_trials,
    )


if __name__ == '__main__':
    main()
