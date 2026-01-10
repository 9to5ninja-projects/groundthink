#!/bin/bash
# Step 5: Build causal-conv1d from source

# Set environment
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="/usr/local/cuda-12.4/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="8.9"
export FORCE_CUDA=1
export MAX_JOBS=1

# Activate venv
source ~/groundthink_env/bin/activate

echo "=== Environment Check ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(which nvcc)"
nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

echo ""
echo "=== Building causal-conv1d ==="
cd ~/causal-conv1d

python setup.py build_ext --inplace 2>&1 | tee build.log

echo ""
echo "=== Checking for errors ==="
grep -i "error\|failed\|aborted" build.log || echo "Build successful - no errors found"

echo ""
echo "=== Installing ==="
pip install -e .
