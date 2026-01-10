#!/usr/bin/env python3
"""Quick test to verify monitoring tools work during training."""

import sys
import os

# Setup environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load train_v4.py's main code with modified config
if __name__ == '__main__':
    # Modify CONFIG before importing
    import train_v4
    train_v4.CONFIG['max_steps'] = 100
    train_v4.CONFIG['save_interval'] = 1000  # Don't save checkpoint for this test
    
    # Run main
    print("=" * 60)
    print("MONITORING TEST: Running 100 steps")
    print("=" * 60)
    train_v4.main()
