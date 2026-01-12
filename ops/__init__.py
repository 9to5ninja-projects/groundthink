# Import RWKV6 unconditionally
from .cuda_backends import RWKV6Attention, RWKV6_CUDA_AVAILABLE

# Lazy import Mamba2 to avoid mamba_ssm dependency when not needed
# Access via: from ops import Mamba2 (will raise ImportError if mamba_ssm missing)
def __getattr__(name):
    if name == 'Mamba2':
        from .cuda_backends import Mamba2
        return Mamba2
    raise AttributeError(f"module 'ops' has no attribute '{name}'")

__all__ = ['RWKV6Attention', 'Mamba2', 'RWKV6_CUDA_AVAILABLE']
