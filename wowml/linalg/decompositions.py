"""
decompositions.py — Matrix decomposition Python wrappers wired to the C kernel.
"""
from __future__ import annotations
import ctypes
from typing import Tuple

from wowml._C._bindings import CWowmlTensor, DTYPE_FROM_STR, check_status, make_c_shape, require_lib
from wowml.tensor import Tensor


def _alloc_out(shape: Tuple[int, ...], dtype: str) -> Tensor:
    c_lib = require_lib()
    t = CWowmlTensor()
    ndim, c_shape = make_c_shape(shape)
    check_status(c_lib.wowml_tensor_alloc(ctypes.byref(t), int(DTYPE_FROM_STR[dtype]), ndim, c_shape))
    return Tensor(dtype=dtype, _c_tensor=t)


def lu(a: Tensor) -> Tuple[Tensor, Tensor]:
    """LU decomposition with partial pivoting."""
    if a.ndim != 2:
        raise ValueError(f"lu requires 2D matrix, got {a.ndim}D.")
    c_lib = require_lib()
    m, n = a.shape
    lu_out = _alloc_out((m, n), a.dtype)
    piv_arr = (ctypes.c_int64 * min(m, n))()
    check_status(c_lib.wowml_linalg_lu(lu_out.c_ptr, piv_arr, a.c_ptr))
    piv_t = Tensor(list(piv_arr), dtype="int64")
    return lu_out, piv_t


def qr(a: Tensor) -> Tuple[Tensor, Tensor]:
    """QR decomposition: A = Q @ R."""
    if a.ndim != 2:
        raise ValueError(f"qr requires 2D matrix, got {a.ndim}D.")
    c_lib = require_lib()
    m, n = a.shape
    k = min(m, n)
    q = _alloc_out((m, k), a.dtype)
    r = _alloc_out((k, n), a.dtype)
    check_status(c_lib.wowml_linalg_qr(q.c_ptr, r.c_ptr, a.c_ptr))
    return q, r


def svd(a: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Singular Value Decomposition: A = U @ diag(S) @ V^T."""
    if a.ndim != 2:
        raise ValueError(f"svd requires 2D matrix, got {a.ndim}D.")
    c_lib = require_lib()
    m, n = a.shape
    k = min(m, n)
    u = _alloc_out((m, m), a.dtype)
    s = _alloc_out((k,), a.dtype)
    vt = _alloc_out((n, n), a.dtype)
    check_status(c_lib.wowml_linalg_svd(u.c_ptr, s.c_ptr, vt.c_ptr, a.c_ptr))
    return u, s, vt


def cholesky(a: Tensor) -> Tensor:
    """Cholesky decomposition (lower triangular): A = L @ L^T."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"cholesky requires square 2D matrix, got {a.shape}.")
    c_lib = require_lib()
    l_out = _alloc_out(a.shape, a.dtype)
    check_status(c_lib.wowml_linalg_cholesky(l_out.c_ptr, a.c_ptr))
    return l_out


def eigh(a: Tensor) -> Tuple[Tensor, Tensor]:
    """Eigenvalue decomposition for symmetric matrices: A = V @ diag(w) @ V^T."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"eigh requires square 2D matrix, got {a.shape}.")
    c_lib = require_lib()
    n = a.shape[0]
    w = _alloc_out((n,), a.dtype)
    v = _alloc_out((n, n), a.dtype)
    check_status(c_lib.wowml_linalg_eigh(w.c_ptr, v.c_ptr, a.c_ptr))
    return w, v
