"""
norms.py — Norms, determinant, and trace wired to the C kernel.
"""
from __future__ import annotations
import ctypes

from wowml._C._bindings import check_status, require_lib
from wowml.tensor import Tensor


def norm(a: Tensor, ord: int = 2) -> float:
    """Compute matrix or vector norm."""
    c_lib = require_lib()
    res = ctypes.c_double()
    check_status(c_lib.wowml_linalg_norm(ctypes.byref(res), a.c_ptr, int(ord)))
    return float(res.value)


def det(a: Tensor) -> float:
    """Compute determinant of a square matrix."""
    c_lib = require_lib()
    res = ctypes.c_double()
    check_status(c_lib.wowml_linalg_det(ctypes.byref(res), a.c_ptr))
    return float(res.value)


def trace(a: Tensor) -> float:
    """Compute matrix trace."""
    c_lib = require_lib()
    res = ctypes.c_double()
    check_status(c_lib.wowml_linalg_trace(ctypes.byref(res), a.c_ptr))
    return float(res.value)
