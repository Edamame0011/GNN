from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

module_name = 'flash_schnet_ext'
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name=module_name,
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name=module_name, # これが C++側の TORCH_EXTENSION_NAME になります
            sources=[
                os.path.join(current_dir, 'binding.cpp'),
                os.path.join(current_dir, 'src', 'functions.cpp'),
                os.path.join(current_dir, 'src', 'kernels.cu'),
            ],
            include_dirs=[
                os.path.join(current_dir, 'include')
            ], 
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)