"""Quick inspection of PIPELINE.generate"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"

import torch  # Must import torch first!
import inspect
from rwkv.utils import PIPELINE

sig = inspect.signature(PIPELINE.generate)
print("PIPELINE.generate parameters:")
for name, param in sig.parameters.items():
    default = param.default if param.default != inspect.Parameter.empty else "(required)"
    print(f"  {name}: {default}")

print("\n--- Source code snippet ---")
source = inspect.getsource(PIPELINE.generate)
# Print first 100 lines
for i, line in enumerate(source.split('\n')[:100]):
    print(line)
