from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Create source list
sources = [
    'ops/csrc/selective_scan.cpp',
    'ops/csrc/selective_scan_kernel.cu'
]

setup(
    name='selective_scan_cuda',
    ext_modules=[
        CUDAExtension(
            name='selective_scan_cuda',
            sources=sources,
            extra_cuda_cflags=['-O3']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
