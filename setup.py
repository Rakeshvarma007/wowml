"""
Build script for wowml.
Compiles the C kernel via CMake and installs the shared library
alongside the Python package.
"""
import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def _lib_filename() -> str:
    if sys.platform == "win32":
        return "wowml_c.dll"
    if sys.platform == "darwin":
        return "wowml_c.dylib"
    return "wowml_c.so"


class CMakeBuild(build_ext):
    """Custom build command that invokes CMake to compile the C kernel."""

    def build_extension(self, ext):
        build_dir = os.path.join(self.build_temp, "cmake_build")
        os.makedirs(build_dir, exist_ok=True)
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        os.makedirs(ext_dir, exist_ok=True)

        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTS=OFF",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={ext_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        build_args = ["--config", "Release"]

        subprocess.check_call(["cmake", os.path.abspath(".")] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_dir)

    def get_outputs(self):
        return [
            os.path.join(os.path.abspath(self.build_lib), "wowml", "_C", _lib_filename())
        ]

    def copy_extensions_to_source(self):
        lib_name = _lib_filename()
        src = os.path.join(self.build_lib, "wowml", "_C", lib_name)
        if os.path.isfile(src):
            dst_dir = os.path.join("wowml", "_C")
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, lib_name))


setup(
    ext_modules=[Extension("wowml._C._wowml_c", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
)
