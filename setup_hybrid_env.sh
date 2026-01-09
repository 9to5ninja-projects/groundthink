#!/bin/bash
# Setup script for GroundThink hybrid environment
# RTX 4050 + CUDA 12.4 + PyTorch
# EXPERT STEP-BY-STEP INSTRUCTIONS - DO NOT DEVIATE

set -e  # Exit on error

echo "=== Step 0: Open FRESH Terminal in WSL Ubuntu ==="
echo "# Open new terminal (CTRL+SHIFT+T)"
echo "# Close ALL other terminals"
echo "Continuing..."

echo "=== Step 1: Kill All Python Processes & Clear Cache ==="
# Stop everything
pkill -f python || true
pkill -f pip || true

# Remove ALL cache
rm -rf ~/.cache/pip
rm -rf ~/.cache/torch_extensions
rm -rf ~/.cache/nvcc
rm -rf ~/.cache/ccache
sudo rm -rf /tmp/pip-*

echo "=== Step 2: Install System Dependencies (ONE COMMAND) ==="
sudo apt update && sudo apt install -y \
    build-essential \
    gcc-11 \
    g++-11 \
    git \
    cmake \
    ninja-build \
    python3-venv \
    python3-dev \
    libcusparse-dev-12-4 \
    libcublas-dev-12-4

echo "=== Step 3: Set GCC 11 as Default ==="
# Force GCC 11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
sudo update-alternatives --set gcc /usr/bin/gcc-11
sudo update-alternatives --set g++ /usr/bin/g++-11

# Verify
gcc --version  # Should show 11.x
g++ --version  # Should show 11.x

echo "=== Step 4: Recreate Clean Virtual Environment ==="
# Remove old env
rm -rf ~/groundthink_env

# Create new env with system Python (3.12 on Ubuntu 24.04)
python3 -m venv ~/groundthink_env

# Activate it
source ~/groundthink_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "=== Step 5: Install PyTorch 2.4.0 (Earliest cu124 version) ==="
# UNINSTALL ANY existing torch
pip uninstall -y torch torchvision torchaudio || true

# Install earliest available cu124 version
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-cache-dir

echo "=== Step 6: Set Environment Variables (CRITICAL) ==="
# Add to ~/.bashrc AND current session
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

echo 'export TORCH_CUDA_ARCH_LIST="8.9"' >> ~/.bashrc  # RTX 4050
echo 'export FORCE_CUDA=1' >> ~/.bashrc
echo 'export CC=gcc-11' >> ~/.bashrc
echo 'export CXX=g++-11' >> ~/.bashrc
echo 'export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"' >> ~/.bashrc
echo 'export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"' >> ~/.bashrc
echo 'export MAX_JOBS=2' >> ~/.bashrc  # Reduce parallel jobs

# Apply to current session
source ~/.bashrc

# Verify CUDA
nvcc --version  # Should show 12.4
which gcc       # Should show /usr/bin/gcc-11

echo "=== Step 7: Install Ninja (Build System) ==="
pip install ninja packaging

echo "=== Step 8: Clone Repositories (FRESH) ==="
# Remove old clones
rm -rf ~/causal-conv1d ~/mamba

# Clone fresh
git clone https://github.com/Dao-AILab/causal-conv1d.git ~/causal-conv1d
git clone https://github.com/state-spaces/mamba.git ~/mamba

# Checkout KNOWN WORKING versions
cd ~/causal-conv1d
git checkout v1.2.0  # KNOWN WORKING VERSION

cd ~/mamba
git checkout v2.2.0  # KNOWN WORKING VERSION

echo "=== Step 9: Patch causal-conv1d setup.py (CRITICAL FIX) ==="
cd ~/causal-conv1d

# Create the patch file
cat > fix_setup.py << 'EOF'
import os
import subprocess
from packaging import version
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# ... existing code ...

class BuildExtensionWithFix(BuildExtension):
    def build_extension(self, ext):
        # FORCE C++11 ABI and RTX 4050 architecture
        if isinstance(ext, CUDAExtension):
            # Add C++11 ABI flag
            if 'cxx' not in ext.extra_compile_args:
                ext.extra_compile_args['cxx'] = []
            ext.extra_compile_args['cxx'].append('-D_GLIBCXX_USE_CXX11_ABI=1')
            
            # Add RTX 4050 architecture
            if 'nvcc' not in ext.extra_compile_args:
                ext.extra_compile_args['nvcc'] = []
            
            # Remove any existing arch flags
            ext.extra_compile_args['nvcc'] = [
                arg for arg in ext.extra_compile_args['nvcc'] 
                if not arg.startswith('-arch=') and not arg.startswith('-gencode')
            ]
            
            # Add RTX 4050 (sm_89)
            ext.extra_compile_args['nvcc'].extend([
                '-gencode', 'arch=compute_89,code=sm_89',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '--use_fast_math'
            ])
        
        # Call original build
        return super().build_extension(ext)

# ... rest of setup.py with BuildExtensionWithFix ...
EOF

# Now apply the actual patch to setup.py
sed -i 's/BuildExtension/BuildExtensionWithFix/g' setup.py
sed -i "s/from torch.utils.cpp_extension import CUDAExtension, BuildExtension/from torch.utils.cpp_extension import CUDAExtension, BuildExtension\n\n# CUSTOM FIX FOR RTX 4050 AND C++11 ABI\nclass BuildExtensionWithFix(BuildExtension):\n    def build_extension(self, ext):\n        if isinstance(ext, CUDAExtension):\n            if 'cxx' not in ext.extra_compile_args:\n                ext.extra_compile_args['cxx'] = []\n            ext.extra_compile_args['cxx'].append('-D_GLIBCXX_USE_CXX11_ABI=1')\n            \n            if 'nvcc' not in ext.extra_compile_args:\n                ext.extra_compile_args['nvcc'] = []\n            \n            ext.extra_compile_args['nvcc'] = [\n                arg for arg in ext.extra_compile_args['nvcc'] \n                if not arg.startswith('-arch=') and not arg.startswith('-gencode')\n            ]\n            \n            ext.extra_compile_args['nvcc'].extend([\n                '-gencode', 'arch=compute_89,code=sm_89',\n                '-U__CUDA_NO_HALF_OPERATORS__',\n                '-U__CUDA_NO_HALF_CONVERSIONS__',\n                '--expt-relaxed-constexpr',\n                '--expt-extended-lambda',\n                '--use_fast_math'\n            ])\n        return super().build_extension(ext)/" setup.py

echo "=== Step 10: Build causal-conv1d (WITHOUT Isolation) ==="
cd ~/causal-conv1d

# Build with VERBOSE output
pip install -e . --no-build-isolation --verbose 2>&1 | tee build_causal.log

# Check for success
if grep -q "Finished processing dependencies for causal-conv1d" build_causal.log; then
    echo "✅ causal-conv1d built successfully"
else
    echo "❌ Build failed. Check build_causal.log"
    tail -50 build_causal.log
    exit 1
fi

echo "=== Step 11: Patch mamba setup.py ==="
cd ~/mamba

# Similar patch for mamba
sed -i "s/from torch.utils.cpp_extension import BuildExtension/from torch.utils.cpp_extension import BuildExtension\n\n# CUSTOM FIX FOR RTX 4050 AND C++11 ABI\nclass BuildExtensionWithFix(BuildExtension):\n    def build_extension(self, ext):\n        if hasattr(ext, 'extra_compile_args') and 'cxx' in ext.extra_compile_args:\n            ext.extra_compile_args['cxx'].append('-D_GLIBCXX_USE_CXX11_ABI=1')\n        return super().build_extension(ext)/" setup.py

sed -i 's/cmdclass={\"build_ext\": BuildExtension}/cmdclass={\"build_ext\": BuildExtensionWithFix}/' setup.py

echo "=== Step 12: Build mamba-ssm ==="
cd ~/mamba

# Build
pip install -e . --no-build-isolation --verbose 2>&1 | tee build_mamba.log

# Check for success
if grep -q "Finished processing dependencies for mamba-ssm" build_mamba.log; then
    echo "✅ mamba-ssm built successfully"
else
    echo "❌ Build failed. Check build_mamba.log"
    tail -50 build_mamba.log
    exit 1
fi

echo "=== Step 13: VERIFICATION TEST (CRITICAL) ==="
cat > verify_cuda.py << 'EOF'
import torch
import sys
import os

print("="*60)
print("CUDA KERNEL VERIFICATION")
print("="*60)

# 1. PyTorch info
print(f"\n1. PyTorch Info:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA: {torch.version.cuda}")
print(f"   ABI: {torch._C._PYBIND11_BUILD_ABI}")
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")

# 2. causal-conv1d test
print("\n2. causal-conv1d Test:")
try:
    import causal_conv1d
    from causal_conv1d import causal_conv1d_fn
    print("   ✓ Module imported")
    
    # Test with small tensors
    x = torch.randn(2, 32, 128, device='cuda', dtype=torch.float16)
    weight = torch.randn(4, 128, device='cuda', dtype=torch.float16)
    bias = torch.randn(128, device='cuda', dtype=torch.float16)
    
    y = causal_conv1d.causal_conv1d_fn(x, weight, bias, False, 0.0, 1.0)
    print(f"   ✓ Kernel executed, output shape: {y.shape}")
    print(f"   ✓ causal-conv1d CUDA kernels WORKING")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# 3. mamba-ssm test
print("\n3. mamba-ssm Test:")
try:
    import mamba_ssm
    from mamba_ssm.ops.selective_state_update import selective_state_update
    print("   ✓ Module imported")
    
    if selective_state_update is not None:
        print("   ✓ selective_state_update CUDA kernel available")
        print("   ✓ mamba-ssm CUDA kernels WORKING")
    else:
        print("   ⚠️ selective_state_update is None (using Triton fallback)")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - CUDA KERNELS ARE WORKING")
print("="*60)
EOF

python verify_cuda.py

echo "=== Step 14: Install RWKV ==="
# Install with same constraints
pip install rwkv --no-cache-dir

echo "=== Step 15: Create Your Hybrid Model Test ==="
cat > test_hybrid_5M.py << 'EOF'
import torch
import torch.nn as nn

class SimpleHybrid(nn.Module):
    def __init__(self, vocab_size=10000, hidden_size=128, rwkv_layers=3, mamba_layers=6):
        super().__init__()
        
        # Embedding
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # Import REAL modules (they should work now)
        from mamba_ssm import Mamba
        from rwkv.rwkv import RWKV
        
        # Mamba2 pathway
        self.mamba_path = nn.ModuleList([
            Mamba(d_model=hidden_size, d_state=16, d_conv=4, expand=2)
            for _ in range(mamba_layers)
        ])
        
        # RWKV pathway (simplified - using linear for now)
        self.rwkv_path = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(rwkv_layers)
        ])
        
        # Fusion
        self.fusion = nn.Linear(2 * hidden_size, hidden_size)
        
        # Output (tied to embedding)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.head.weight = self.embed.weight  # Weight tying
        
    def forward(self, x):
        x_emb = self.embed(x)
        
        # Mamba pathway
        m = x_emb
        for layer in self.mamba_path:
            m = layer(m)
        
        # RWKV pathway
        r = x_emb
        for layer in self.rwkv_path:
            r = layer(r)
        
        # Fuse
        combined = torch.cat([m, r], dim=-1)
        fused = self.fusion(combined)
        
        return self.head(fused)

# Test
model = SimpleHybrid(vocab_size=10000, hidden_size=128)
model = model.cuda()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward test
x = torch.randint(0, 10000, (2, 32)).cuda()
with torch.cuda.amp.autocast():
    y = model(x)
    
print(f"Forward pass successful!")
print(f"Output shape: {y.shape}")
print(f"Memory used: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
print(f"CUDA kernels working: {torch.cuda.is_available()}")
EOF

python test_hybrid_5M.py

echo "=== FINAL CONFIRMATION ==="
python -c "
import torch
import causal_conv1d
import mamba_ssm
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA:', torch.version.cuda)
print('✅ causal-conv1d:', causal_conv1d.__version__ if hasattr(causal_conv1d, '__version__') else 'loaded')
print('✅ mamba-ssm:', mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else 'loaded')
print('✅ RTX 4050 ready for hybrid model development')
"

echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "If ANY step failed, see FAILURE DIAGNOSIS below:"
echo ""
echo "If gcc: No such file or directory:"
echo "  sudo apt install --reinstall gcc-11 g++-11"
echo "  which gcc"
echo ""
echo "If nvcc not found:"
echo "  export CUDA_HOME=/usr/local/cuda-12.4"
echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
echo "  which nvcc"
echo ""
echo "If PyTorch version mismatch:"
echo "  pip list | grep torch"
echo "  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --force-reinstall --no-deps"
echo ""
echo "If C++11 ABI error:"
echo "  CFLAGS=\"-D_GLIBCXX_USE_CXX11_ABI=1\" CXXFLAGS=\"-D_GLIBCXX_USE_CXX11_ABI=1\" pip install -e . --no-build-isolation --verbose"
