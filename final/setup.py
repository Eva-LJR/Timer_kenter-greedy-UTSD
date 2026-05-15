from pathlib import Path
import os
import shutil
import subprocess

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


class BuildExtWithCuda(build_ext):
    def build_extensions(self):
        nvcc = shutil.which("nvcc")
        for ext in self.extensions:
            if ext.name != "twed_kcenter":
                continue
            if nvcc:
                self._build_cuda_object(ext, nvcc)
            else:
                print("[setup.py] nvcc not found, building CPU-only extension.")
        super().build_extensions()

    def _build_cuda_object(self, ext, nvcc):
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        obj_ext = ".obj" if self.compiler.compiler_type == "msvc" else ".o"
        cuda_obj = build_temp / f"twed_cuda_kernel{obj_ext}"

        include_dirs = list(ext.include_dirs or [])

        cmd = [
            nvcc,
            "-c",
            "twed_cuda_kernel.cu",
            "-o",
            str(cuda_obj),
            "-O3",
            "--std=c++14",
        ]

        for inc in include_dirs:
            cmd.extend(["-I", inc])

        if self.compiler.compiler_type == "msvc":
            cmd.extend(["-Xcompiler", "/MD,/utf-8"])
        else:
            cmd.extend(["-Xcompiler", "-fPIC"])

        print("[setup.py] Building CUDA object:", " ".join(cmd))
        subprocess.check_call(cmd)

        ext.extra_objects = list(getattr(ext, "extra_objects", [])) + [str(cuda_obj)]
        ext.define_macros = list(getattr(ext, "define_macros", [])) + [("WITH_CUDA", "1")]

        ext.libraries = list(getattr(ext, "libraries", [])) + ["cudart"]
        ext.library_dirs = list(getattr(ext, "library_dirs", []))

        if self.compiler.compiler_type == "msvc":
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                ext.library_dirs.append(os.path.join(cuda_path, "lib", "x64"))
            else:
                print("[setup.py] WARNING: CUDA_PATH not set; cudart link may fail on Windows.")
        else:
            ext.library_dirs.append("/usr/local/cuda/lib64")


extra_compile_args = ["-O3", "-ffast-math"]

if os.name == "nt":
    extra_compile_args = ["/O2", "/utf-8"]

ext_modules = [
    Pybind11Extension(
        "twed_kcenter",
        ["twed_kcenter_module.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="twed_kcenter",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithCuda},
)
