from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='twed_cuda',
    ext_modules=[
        CUDAExtension('twed_cuda', [
            'twed_cuda.cpp',
            'twed_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension},
)