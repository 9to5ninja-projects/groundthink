"""
GroundThink Model Configuration

Based on RWKV-7 architecture with enhanced grounding mechanisms.
RWKV-7 already has input-dependent selective decay (similar to Mamba's S6),
so the innovation is in the routing and grounding layers.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class GroundThinkConfig:
    """Configuration for GroundThink hybrid model"""
    
    # Model dimensions
    d_model: int = 2560
    n_layers: int = 32
    n_heads: int = 40
    head_dim: int = 64
    state_dim: int = 64
    ffn_expansion: float = 3.5
    vocab_size: int = 65536  # RWKV World vocab
    
    # Selection mechanism (RWKV-7 already has this)
    # These control the input-dependent decay
    selection_rank: int = 160
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: Literal["random", "constant"] = "random"
    dt_scale: float = 1.0
    
    # Grounding mechanism (NEW - prevents state drift)
    use_grounding: bool = True
    grounding_conv_kernel: int = 4
    min_retention: float = 0.01  # Never fully forget
    max_retention: float = 0.99  # Always some decay
    
    # Normalization
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    norm_eps: float = 1e-6
    
    # Initialization
    w_init: Literal["power_law", "uniform"] = "power_law"  # Faster decay in early layers
    output_init: Literal["zero", "small"] = "zero"  # O3 initialization
    dt_bias: float = 1.0  # Start with moderate memory
    
    # Training
    max_seq_len: int = 131072
    chunk_size: int = 256  # For chunked parallel scan
    
    # Hardware
    use_triton: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"


# Preset configurations
GROUNDTHINK_160M = GroundThinkConfig(
    d_model=768,
    n_layers=12,
    n_heads=12,
    head_dim=64,
    state_dim=32,
    ffn_expansion=3.5,
)

GROUNDTHINK_1_6B = GroundThinkConfig(
    d_model=2048,
    n_layers=24,
    n_heads=32,
    head_dim=64,
    state_dim=64,
    selection_rank=128,
)

GROUNDTHINK_2_8B = GroundThinkConfig(
    d_model=2560,
    n_layers=32,
    n_heads=40,
    head_dim=64,
    state_dim=64,
    selection_rank=160,
)
