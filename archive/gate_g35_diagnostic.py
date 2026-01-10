"""
Gate G3.5 Diagnostic: State Health Check

Per senior guidance, verify that:
1. State norm fluctuates over time (not static at 256)
2. State matrix has low rank (structured, not random noise)
3. State values aren't saturating gates

SUCCESS THRESHOLDS (Gate G3.5):
+---------------------------+-------------+-------------+----------------------------------+
| Metric                    | Pass        | Warn        | Fail                             |
+---------------------------+-------------+-------------+----------------------------------+
| Norm Std (over 512 tok)   | > 1.0       | 0.1 - 1.0   | < 0.1 (static/frozen)            |
| SVD Top-1 Ratio           | > 0.3       | 0.15 - 0.3  | < 0.15 (noise)                   |
| SVD Top-5 Ratio           | > 0.5       | 0.3 - 0.5   | < 0.3 (high rank/noise)          |
| Saturation (|x| > 5.0)    | < 10%       | 10% - 30%   | > 30% (gates stuck)              |
+---------------------------+-------------+-------------+----------------------------------+

FAILURE INTERPRETATION:
| Failure         | Meaning                              | Senior Fix                        |
|-----------------|--------------------------------------|-----------------------------------|
| High Rank       | Model doesn't know what to remember  | Increase Gating LR on B/C matrices|
| Static Norm     | State is leaking or resetting wrong  | Check State-Handoff implementation|
| High Saturation | Internal energy too high             | Lower gamma from 0.01 to 0.001    |
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import math

from layers_v030 import GroundThinkV3
from data_v030 import load_stateful_dataset

def main():
    # Load model and data
    dataset, tokenizer = load_stateful_dataset('shakespeare.txt', 1, 512)
    
    model = GroundThinkV3(
        vocab_size=tokenizer.vocab_size,
        n_layers=12, dim=256, n_heads=8, head_dim=32, state_dim=16,
        attn_positions=[6],
    ).cuda()
    
    # Load trained checkpoint
    ckpt = torch.load('groundthink_8M_v3_1k.pt', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    x, y, _ = dataset[0]
    x = x.cuda()
    
    # Get attention layer positions to skip
    attn_positions = getattr(model.stack, 'attn_positions', [model.n_layers // 2])
    if attn_positions is None:
        attn_positions = [model.n_layers // 2]
    
    print('='*60)
    print('CHECK A: Per-Layer State COSINE SIMILARITY Over Time (512 tokens)')
    print('='*60)
    print(f'Measuring cosine similarity between consecutive states')
    print(f'Attention layers at {attn_positions} will return None (expected)')
    print(f'NOTE: Norm variance is useless due to StateNorm (see V3.5 Section 9.1)\n')
    
    # Track cosine similarities per layer over time
    # cosine_sim = 1.0 means no change, < 1.0 means state is evolving
    cos_sims_by_layer = {i: [] for i in range(model.n_layers)}
    prev_states = [None] * model.n_layers
    
    state = None
    with torch.no_grad():
        for t in range(min(512, x.shape[1])):
            tok = x[:, t:t+1]
            logits, state = model(tok, state)
            
            for layer_idx, layer_state in enumerate(state):
                if layer_state is not None:
                    # Flatten state for cosine similarity
                    flat_state = layer_state.view(-1)
                    
                    if prev_states[layer_idx] is not None:
                        # Compute cosine similarity with previous state
                        prev_flat = prev_states[layer_idx]
                        cos_sim = torch.nn.functional.cosine_similarity(
                            flat_state.unsqueeze(0), 
                            prev_flat.unsqueeze(0)
                        ).item()
                        cos_sims_by_layer[layer_idx].append(cos_sim)
                    
                    # Store current state for next iteration
                    prev_states[layer_idx] = flat_state.clone()
    
    # Compute per-layer cosine similarity statistics
    print('--- Per-Layer Cosine Similarity (consecutive states) ---')
    print('cos_sim=1.0 means STATIC, <1.0 means EVOLVING')
    print()
    
    layer_cos_vars = {}
    worst_mean_cos = 0.0  # Track highest (most static) - closer to 1.0 is worse
    worst_layer = 0
    
    for layer_idx in range(model.n_layers):
        if layer_idx in attn_positions:
            print(f'Layer {layer_idx:2d}: [ATTENTION - returns None]')
            continue
        
        cos_sims = cos_sims_by_layer[layer_idx]
        if len(cos_sims) > 0:
            mean_cos = torch.tensor(cos_sims).mean().item()
            std_cos = torch.tensor(cos_sims).std().item()
            min_cos = min(cos_sims)
            max_cos = max(cos_sims)
            
            layer_cos_vars[layer_idx] = std_cos
            
            # Flag static layers (mean > 0.99 = barely changing)
            if mean_cos > 0.999:
                status = '[FROZEN]'
            elif mean_cos > 0.99:
                status = '[STATIC]'
            elif mean_cos > 0.9:
                status = '[EVOLVING]'
            else:
                status = '[DYNAMIC]'
            
            print(f'Layer {layer_idx:2d}: mean={mean_cos:.6f}, std={std_cos:.6f}, range=[{min_cos:.4f}, {max_cos:.4f}] {status}')
            
            # Track the most static layer (highest mean_cos)
            if mean_cos > worst_mean_cos:
                worst_mean_cos = mean_cos
                worst_layer = layer_idx
    
    # Overall assessment
    print(f'\n--- Cosine Similarity Analysis ---')
    print(f'Most static layer: Layer {worst_layer} with mean_cos={worst_mean_cos:.6f}')
    
    # Check for "Identity Coalescence" pattern (late layers should be more dynamic)
    early_means = []
    late_means = []
    for layer_idx, cos_sims in cos_sims_by_layer.items():
        if layer_idx in attn_positions or len(cos_sims) == 0:
            continue
        mean_cos = torch.tensor(cos_sims).mean().item()
        if layer_idx < 6:
            early_means.append(mean_cos)
        elif layer_idx > 6:
            late_means.append(mean_cos)
    
    early_avg = sum(early_means) / len(early_means) if early_means else 1.0
    late_avg = sum(late_means) / len(late_means) if late_means else 1.0
    
    print(f'Early layers (0-5) avg cosine: {early_avg:.6f}')
    print(f'Late layers (7-11) avg cosine: {late_avg:.6f}')
    
    # Lower cosine = more dynamic
    if late_avg < early_avg - 0.01:
        print('[OK] Late layers more dynamic - identity may be coalescing')
    elif early_avg < late_avg - 0.01:
        print('[WARN] Early layers more dynamic - signal dying in later layers')
    else:
        print('[WARN] Similar dynamics across stack')
    
    # Pass/fail based on worst case
    # Pass: at least some layers are evolving (mean_cos < 0.999)
    if worst_mean_cos > 0.999:
        print('\n[FAIL] FROZEN STATE DETECTED: All layers have cosine > 0.999')
        state_healthy = False
    elif worst_mean_cos > 0.99:
        print('\n[WARN] STATIC STATE WARNING: Some layers barely evolving')
        state_healthy = False
    else:
        print('\n[OK] States are evolving - model is processing information')
        state_healthy = True
    
    print('\n' + '='*60)
    print('CHECK B: SVD Rank of State Matrix (RECURRENT layers only)')
    print('='*60)
    
    # Get final state after full sequence
    with torch.no_grad():
        logits, final_state = model(x, None)
    
    # Get attention layer positions from the model
    # Default is middle layer (layer 6 for 12 layers)
    attn_positions = getattr(model.stack, 'attn_positions', [model.n_layers // 2])
    if attn_positions is None:
        attn_positions = [model.n_layers // 2]
    print(f'NOTE: Skipping attention layers at positions: {attn_positions}')
    print('      (Attention layers pass state unchanged, expected high rank)\n')
    
    # Collect worst-case metrics across RECURRENT layers only
    n_layers = len(final_state)
    worst_top5_ratio = 1.0
    worst_layer_rank = 0
    worst_head_rank = 0
    
    all_layer_stats = []
    
    for layer_idx in range(n_layers):
        # Skip attention layers - they don't update state
        if layer_idx in attn_positions:
            print(f'Layer {layer_idx}: [ATTENTION - skipped]')
            continue
            
        layer_state = final_state[layer_idx]  # [B, H, D, D]
        n_heads = layer_state.shape[1]
        
        layer_worst_ratio = 1.0
        for head_idx in range(n_heads):
            state_matrix = layer_state[0, head_idx, :, :]  # [32, 32]
            
            # Compute SVD
            U, S, V = torch.linalg.svd(state_matrix)
            
            top5_ratio = S[:5].sum().item() / (S.sum().item() + 1e-10)
            
            all_layer_stats.append({
                'layer': layer_idx,
                'head': head_idx,
                'top5_ratio': top5_ratio,
                'top_sv': S[0].item()
            })
            
            if top5_ratio < layer_worst_ratio:
                layer_worst_ratio = top5_ratio
            
            if top5_ratio < worst_top5_ratio:
                worst_top5_ratio = top5_ratio
                worst_layer_rank = layer_idx
                worst_head_rank = head_idx
        
        print(f'Layer {layer_idx}: worst head top-5 ratio = {layer_worst_ratio:.3f}')
    
    print(f'\n--- WORST CASE (recurrent layers only) ---')
    print(f'Worst top-5 ratio: {worst_top5_ratio:.3f} (Layer {worst_layer_rank}, Head {worst_head_rank})')
    
    top5_ratio = worst_top5_ratio  # Use worst case for final verdict
    
    if top5_ratio > 0.8:
        print('[OK] All recurrent heads have low effective rank - state is structured/compressed')
    elif top5_ratio > 0.5:
        print('[WARN] Some heads have medium rank - some structure but not optimal')
    else:
        print('[FAIL] Some heads have high rank - state looks like noise, not learning features')
    
    print('\n' + '='*60)
    print('CHECK C: Gate Saturation (RECURRENT layers only)')
    print('='*60)
    
    # Check saturation across RECURRENT layers only
    worst_saturation = 0.0
    worst_layer_sat = 0
    worst_head_sat = 0
    
    for layer_idx in range(n_layers):
        # Skip attention layers
        if layer_idx in attn_positions:
            continue
            
        layer_state = final_state[layer_idx]  # [B, H, D, D]
        n_heads = layer_state.shape[1]
        
        for head_idx in range(n_heads):
            state_matrix = layer_state[0, head_idx, :, :]
            saturated_pct = (state_matrix.abs() > 5.0).float().mean().item()
            
            if saturated_pct > worst_saturation:
                worst_saturation = saturated_pct
                worst_layer_sat = layer_idx
                worst_head_sat = head_idx
    
    print(f'Worst saturation: {worst_saturation*100:.1f}% (Layer {worst_layer_sat}, Head {worst_head_sat})')
    
    saturated = worst_saturation  # Use worst case for final verdict
    
    if saturated > 0.3:
        print('[FAIL] High saturation - gates are stuck, model cannot be selective')
    elif saturated > 0.1:
        print('[WARN] Moderate saturation - may impact learning')
    else:
        print('[OK] Low saturation - gates can operate normally')
    
    print('\n' + '='*60)
    print('GATE G3.5 SUMMARY')
    print('='*60)
    
    issues = []
    if not state_healthy:
        issues.append('Static/frozen state (cosine similarity > 0.99)')
    if top5_ratio < 0.5:
        issues.append('High rank state (noise-like, not structured)')
    if saturated > 0.3:
        issues.append('Gate saturation (selective mechanism broken)')
    
    if issues:
        print('[FAIL] GATE G3.5 FAILED - Issues found:')
        for issue in issues:
            print(f'   - {issue}')
        print('\nDo NOT proceed to G4 until these are resolved.')
    else:
        print('[PASS] GATE G3.5 PASSED - State appears healthy')
        print('   Ready to proceed to Phase 4 Evaluation')


if __name__ == '__main__':
    main()
