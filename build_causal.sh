#!/bin/bash
# Set CUDA 12.4 environment BEFORE activating venv
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="/usr/local/cuda-12.4/bin:${PATH}"

# Activate venv
source ~/groundthink_env/bin/activate

echo "=== Environment Check ==="
echo "Python: $(which python)"
echo "pip: $(which pip)"
echo "nvcc: $(which nvcc)"
nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

echo ""
echo "=== Building causal-conv1d ==="
TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=2 pip install --no-cache-dir --no-build-isolation causal-conv1d -v 2>&1 | tail -100
