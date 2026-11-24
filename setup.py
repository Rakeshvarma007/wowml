"""
Build script for wowml.
Compiles the C kernel via CMake and installs the shared library
alongside the Python package.
"""
import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    """Custom build command that invokes CMake to compile the C kernel."""

    def build_extension(self, ext):
        build_dir = os.path.join(self.build_temp, "cmake_build")
        os.makedirs(build_dir, exist_ok=True)
        install_dir = os.path.abspath(self.build_lib)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={install_dir}/wowml/_C",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        build_args = ["--config", "Release"]

        subprocess.check_call(["cmake", os.path.abspath(".")] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)


setup(
    ext_modules=[Extension("wowml._C._wowml_c", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
)
