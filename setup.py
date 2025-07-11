from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

setup(
    name="debug_print",
    version="0.0.2",
    packages=['debug_print'],
    ext_modules=[
        CUDAExtension(
            name="debug_print._kernels",
            sources=["csrc/debug_print.cu"],
            extra_compile_args={"nvcc": ["-O3", "-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
