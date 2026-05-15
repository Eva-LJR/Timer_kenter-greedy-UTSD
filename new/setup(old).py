from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "twed_kcenter",
        ["twed_kcenter_module.cpp"],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    ),
]

setup(
    name="twed_kcenter",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)