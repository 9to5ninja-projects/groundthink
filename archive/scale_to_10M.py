"""
SCALING THE INTELLIGENT HYBRID - Architecture Validation
Following the original design with proper head_dim=64 alignment
"""

import torch
import sys
sys.path.insert(0, '.')

from intelligent_hybrid import (
    IntelligentHybridModel, 
    INTELLIGENT_SCALING, 
    test_model_long_context
)


def analyze_all_scales():
    """
    Analyze all scaling configurations to understand parameter counts
    """
    print("\n" + "="*70)
    print("INTELLIGENT HYBRID SCALING ANALYSIS")
    print("Based on original design: head_dim=64 for Tensor Core alignment")
    print("="*70)
    
    vocab_size = 10000
    
    print(f"\n{'Scale':<10} {'Dim':>6} {'Depth':>6} {'Heads':>6} {'State':>6} {'Params':>12} {'Category'}")
    print("-" * 70)
    
    for scale_name, config in INTELLIGENT_SCALING.items():
        model = IntelligentHybridModel(
            vocab_size=vocab_size,
            dim=config['dim'],
            depth=config['depth'],
            state_dim=config['state_dim']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Categorize
        if total_params < 10_000_000:
            category = "validation"
        elif total_params < 100_000_000:
            category = "training"
        elif total_params < 1_000_000_000:
            category = "production"
        else:
            category = "large-scale"
        
        print(f"{scale_name:<10} {config['dim']:>6} {config['depth']:>6} "
              f"{config['heads']:>6} {config['state_dim']:>6} "
              f"{total_params:>12,} {category}")
        
        del model
    
    print("-" * 70)


def test_validation_scale():
    """
    Test the 'valid' scale - architecture validation before scaling
    """
    print("\n" + "="*70)
    print("TESTING 'valid' SCALE (~15M params)")
    print("This validates the architecture before scaling to production")
    print("="*70)
    
    config = INTELLIGENT_SCALING['valid']
    
    model = IntelligentHybridModel(
        vocab_size=10000,
        dim=config['dim'],
        depth=config['depth'],
        state_dim=config['state_dim']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel Configuration:")
    print(f"  Dimension: {config['dim']}")
    print(f"  Depth: {config['depth']}")
    print(f"  State dimension: {config['state_dim']}")
    print(f"  Heads: {config['heads']}")
    print(f"  Head dimension: {config['dim'] // config['heads']}")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Parameter breakdown
    print(f"\nParameter breakdown:")
    embed_params = model.embed.weight.numel()
    block_params = sum(sum(p.numel() for p in block.parameters()) for block in model.blocks)
    print(f"  Embedding: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
    print(f"  All blocks: {block_params:,} ({block_params/total_params*100:.1f}%)")
    print(f"  Weight tying: {'✅ ON' if model.head.weight is model.embed.weight else '❌ OFF'}")
    
    # Architecture checks
    print(f"\nArchitecture validation:")
    head_dim = config['dim'] // config['heads']
    print(f"  Head dimension = {head_dim} {'✅' if head_dim == 64 else '⚠️ (should be 64)'}")
    print(f"  State matrix per head: {head_dim}×{head_dim} = {head_dim**2} params")
    print(f"  Total state size: {config['heads']} heads × {head_dim**2} = {config['heads'] * head_dim**2}")
    
    return model, config


if __name__ == "__main__":
    # Step 1: Analyze all scales
    analyze_all_scales()
    
    # Step 2: Test validation scale
    model, config = test_validation_scale()
    
    # Step 3: Run long context tests
    print("\n" + "="*70)
    print("RUNNING LONG CONTEXT TESTS")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    test_results = test_model_long_context(model, scale_name='valid (~15M)')
    
    # Step 4: Decision
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)
    
    if test_results['summary'].get('overall_pass', False) or \
       test_results['summary'].get('architecture_sound', False):
        print("✅ ARCHITECTURE VALIDATED")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Max sequence: {test_results['summary'].get('max_supported_length', 'N/A')}")
        print("\n🎯 NEXT STEPS:")
        print("   1. Train on small dataset (validate learning)")
        print("   2. Test context retention after training")
        print("   3. Scale to 'small' (~30M) and repeat")
    else:
        print("❌ ARCHITECTURE NEEDS ADJUSTMENT")
        print("   Review test failures and fix before training")
