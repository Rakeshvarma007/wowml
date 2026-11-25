"""
_bindings.py — ctypes bindings to the compiled wowml_c shared library.
Defines C struct layouts, function signatures, and error status handlers.
"""
from __future__ import annotations
import ctypes
import sys
import pathlib
from typing import Optional


class CWowmlTensor(ctypes.Structure):
    """ctypes representation matching wowml_tensor_t in csrc/include/wowml/tensor.h."""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int64 * 8),
        ("strides", ctypes.c_int64 * 8),
        ("numel", ctypes.c_int64),
        ("owns_data", ctypes.c_int),
    ]


def _find_library() -> pathlib.Path:
    """Locate the compiled wowml_c shared library."""
    here = pathlib.Path(__file__).resolve().parent

    if sys.platform == "win32":
        names = ["wowml_c.dll", "libwowml_c.dll"]
    elif sys.platform == "darwin":
        names = ["wowml_c.dylib", "libwowml_c.dylib"]
    else:
        names = ["wowml_c.so", "libwowml_c.so"]

    search_dirs = [
        here,
        here.parent.parent / "build",
        here.parent.parent / "build" / "Release",
    ]

    for d in search_dirs:
        for name in names:
            p = d / name
            if p.exists():
                return p

    raise OSError("Could not find wowml_c shared library. Build with cmake first.")


def check_status(status_code: int) -> None:
    """Validate status code returned by C kernel and raise exception on error."""
    # TODO: Implement error checking stub
    pass


def _setup_bindings(c_lib: ctypes.CDLL) -> None:
    """Register argtypes and restype on all exported C kernel symbols."""
    # TODO: Register signatures for tensor, memory, ops, fused kernels, matmul, and linalg
    pass


try:
    lib: Optional[ctypes.CDLL] = ctypes.CDLL(str(_find_library()))
    _setup_bindings(lib)
except OSError:
    lib = None
