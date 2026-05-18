from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_graph_ops", 
    ext_modules=[
        CUDAExtension(
            name="my_graph_ops", 
            sources=[
                "calc_force_kernel.cu", 
                "message_agg_kernel.cu", 
                "binding.cpp", 
            ], 
            extra_compile_args={
                "cxx": ["-O3"], 
                "nvcc": ["-O3"]
            }
        )
    ], 
    cmdclass={
        "build_ext": BuildExtension
    }
)