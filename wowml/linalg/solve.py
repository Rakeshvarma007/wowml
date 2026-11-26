"""
solve.py — Linear system solvers wired to the C kernel.
"""
from __future__ import annotations
import ctypes

from wowml._C._bindings import CWowmlTensor, DTYPE_FROM_STR, check_status, make_c_shape, require_lib
from wowml.tensor import Tensor


def solve(a: Tensor, b: Tensor) -> Tensor:
    """Solve linear system Ax = b for x."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"solve requires square 2D A, got {a.shape}.")
    c_lib = require_lib()
    x = CWowmlTensor()
    ndim, c_shape = make_c_shape(b.shape)
    check_status(c_lib.wowml_tensor_alloc(ctypes.byref(x), int(DTYPE_FROM_STR[a.dtype]), ndim, c_shape))
    out = Tensor(dtype=a.dtype, _c_tensor=x)
    check_status(c_lib.wowml_linalg_solve(out.c_ptr, a.c_ptr, b.c_ptr))
    return out


def inv(a: Tensor) -> Tensor:
    """Compute the inverse of a square matrix."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"inv requires square 2D matrix, got {a.shape}.")
    c_lib = require_lib()
    out_c = CWowmlTensor()
    ndim, c_shape = make_c_shape(a.shape)
    check_status(c_lib.wowml_tensor_alloc(ctypes.byref(out_c), int(DTYPE_FROM_STR[a.dtype]), ndim, c_shape))
    out = Tensor(dtype=a.dtype, _c_tensor=out_c)
    check_status(c_lib.wowml_linalg_inv(out.c_ptr, a.c_ptr))
    return out
