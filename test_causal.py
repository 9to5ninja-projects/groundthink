#!/usr/bin/env python3
"""Test causal_conv1d import."""
import sys
try:
    import causal_conv1d_cuda
    print(f"causal_conv1d_cuda: OK - {causal_conv1d_cuda}")
except ImportError as e:
    print(f"causal_conv1d_cuda: FAILED - {e}")
    sys.exit(1)

try:
    from causal_conv1d import causal_conv1d_fn
    print(f"causal_conv1d_fn: OK - {causal_conv1d_fn}")
except ImportError as e:
    print(f"causal_conv1d_fn: FAILED - {e}")
