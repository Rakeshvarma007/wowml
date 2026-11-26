"""
_bindings.py — High-speed ctypes bindings to the compiled wowml_c shared library.
Defines C struct layouts, dtype/status enums, full argtypes/restype signatures,
error translation, and zero-copy NumPy interop helpers.
"""
from __future__ import annotations
import ctypes
import os
import pathlib
import sys
from enum import IntEnum
from typing import Optional, Sequence, Tuple

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for _dll_dir in ("C:/mingw64/bin", "C:/msys64/ucrt64/bin"):
        if os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)

WOWML_MAX_DIMS = 8


class DType(IntEnum):
    FLOAT32 = 0
    FLOAT64 = 1
    INT32 = 2
    INT64 = 3


DTYPE_FROM_STR = {
    "float32": DType.FLOAT32,
    "f32": DType.FLOAT32,
    "float": DType.FLOAT32,
    "float64": DType.FLOAT64,
    "f64": DType.FLOAT64,
    "double": DType.FLOAT64,
    "int32": DType.INT32,
    "i32": DType.INT32,
    "int64": DType.INT64,
    "i64": DType.INT64,
}

DTYPE_TO_STR = {
    DType.FLOAT32: "float32",
    DType.FLOAT64: "float64",
    DType.INT32: "int32",
    DType.INT64: "int64",
}

DTYPE_TO_CTYPE = {
    DType.FLOAT32: ctypes.c_float,
    DType.FLOAT64: ctypes.c_double,
    DType.INT32: ctypes.c_int32,
    DType.INT64: ctypes.c_int64,
}


class Status(IntEnum):
    OK = 0
    ERR_NULL_PTR = 1
    ERR_SHAPE_MISMATCH = 2
    ERR_OUT_OF_MEMORY = 3
    ERR_INVALID_DTYPE = 4
    ERR_SINGULAR_MATRIX = 5
    ERR_INVALID_ARGUMENT = 6
    ERR_NOT_IMPLEMENTED = 7


class CWowmlTensor(ctypes.Structure):
    """ctypes representation matching wowml_tensor_t in csrc/include/wowml/tensor.h."""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int64 * WOWML_MAX_DIMS),
        ("strides", ctypes.c_int64 * WOWML_MAX_DIMS),
        ("numel", ctypes.c_int64),
        ("owns_data", ctypes.c_int),
    ]


class CWowmlArena(ctypes.Structure):
    """ctypes representation matching wowml_arena_t in csrc/include/wowml/memory.h."""
    _fields_ = [
        ("buffer", ctypes.c_void_p),
        ("capacity", ctypes.c_size_t),
        ("offset", ctypes.c_size_t),
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
    """Validate status code returned by C kernel and raise an appropriate Python exception."""
    if status_code == Status.OK:
        return
    msg = "Unknown C kernel error"
    if lib is not None:
        raw = lib.wowml_status_string(status_code)
        if raw:
            msg = raw.decode("utf-8", errors="replace")

    if status_code == Status.ERR_NOT_IMPLEMENTED:
        raise NotImplementedError(f"C kernel: {msg}")
    if status_code == Status.ERR_OUT_OF_MEMORY:
        raise MemoryError(f"C kernel: {msg}")
    if status_code in (Status.ERR_SHAPE_MISMATCH, Status.ERR_INVALID_DTYPE, Status.ERR_INVALID_ARGUMENT):
        raise ValueError(f"C kernel: {msg}")
    if status_code == Status.ERR_SINGULAR_MATRIX:
        raise ValueError(f"C kernel linear algebra error: {msg}")
    raise RuntimeError(f"C kernel error ({status_code}): {msg}")


def require_lib() -> ctypes.CDLL:
    """Return the loaded C library or raise a clear error if not built."""
    if lib is None:
        raise RuntimeError(
            "The wowml_c shared library is not loaded. "
            "Compile it first with: cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release"
        )
    return lib


def make_c_shape(shape: Sequence[int]) -> Tuple[int, ctypes.Array]:
    """Convert a Python shape tuple into (ndim, int64_t* array)."""
    ndim = len(shape)
    arr_type = ctypes.c_int64 * ndim
    return ndim, arr_type(*shape)


def c_tensor_from_numpy(arr) -> CWowmlTensor:
    """
    Create a zero-copy CWowmlTensor view pointing directly to a contiguous NumPy array's buffer.
    The NumPy array must remain alive for the lifetime of the view (owns_data = 0).
    """
    import numpy as np
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    np_dtype_map = {
        np.dtype("float32"): DType.FLOAT32,
        np.dtype("float64"): DType.FLOAT64,
        np.dtype("int32"): DType.INT32,
        np.dtype("int64"): DType.INT64,
    }
    if arr.dtype not in np_dtype_map:
        raise TypeError(f"Unsupported NumPy dtype for WowML: {arr.dtype}")

    c_lib = require_lib()
    t = CWowmlTensor()
    ndim, c_shape = make_c_shape(arr.shape)
    status = c_lib.wowml_tensor_from_data(
        ctypes.byref(t),
        ctypes.c_void_p(arr.ctypes.data),
        int(np_dtype_map[arr.dtype]),
        ndim,
        c_shape,
        None,
        0,
    )
    check_status(status)
    return t


def numpy_from_c_tensor(t: CWowmlTensor):
    """
    Return a NumPy array view over a contiguous CWowmlTensor buffer (zero-copy).
    """
    import numpy as np
    if not t.data:
        raise ValueError("Cannot convert unallocated CWowmlTensor to NumPy array.")
    dtype = DType(t.dtype)
    c_type = DTYPE_TO_CTYPE[dtype]
    shape = tuple(t.shape[i] for i in range(t.ndim))
    ptr = ctypes.cast(t.data, ctypes.POINTER(c_type))
    return np.ctypeslib.as_array(ptr, shape=shape)


def _setup_bindings(c_lib: ctypes.CDLL) -> None:
    """Register argtypes and restype on all exported C kernel symbols."""
    T_PTR = ctypes.POINTER(CWowmlTensor)
    I64_PTR = ctypes.POINTER(ctypes.c_int64)
    DBL_PTR = ctypes.POINTER(ctypes.c_double)

    # ---- error.h ----
    c_lib.wowml_status_string.argtypes = [ctypes.c_int]
    c_lib.wowml_status_string.restype = ctypes.c_char_p

    # ---- dtype.h ----
    c_lib.wowml_dtype_size.argtypes = [ctypes.c_int]
    c_lib.wowml_dtype_size.restype = ctypes.c_size_t
    c_lib.wowml_dtype_name.argtypes = [ctypes.c_int]
    c_lib.wowml_dtype_name.restype = ctypes.c_char_p

    # ---- memory.h ----
    c_lib.wowml_alloc.argtypes = [ctypes.c_size_t]
    c_lib.wowml_alloc.restype = ctypes.c_void_p
    c_lib.wowml_calloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    c_lib.wowml_calloc.restype = ctypes.c_void_p
    c_lib.wowml_free.argtypes = [ctypes.c_void_p]
    c_lib.wowml_free.restype = None
    c_lib.wowml_aligned_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    c_lib.wowml_aligned_alloc.restype = ctypes.c_void_p
    c_lib.wowml_aligned_free.argtypes = [ctypes.c_void_p]
    c_lib.wowml_aligned_free.restype = None

    # ---- tensor.h ----
    c_lib.wowml_tensor_alloc.argtypes = [T_PTR, ctypes.c_int, ctypes.c_int, I64_PTR]
    c_lib.wowml_tensor_alloc.restype = ctypes.c_int
    c_lib.wowml_tensor_free.argtypes = [T_PTR]
    c_lib.wowml_tensor_free.restype = ctypes.c_int
    c_lib.wowml_tensor_from_data.argtypes = [
        T_PTR, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, I64_PTR, I64_PTR, ctypes.c_int
    ]
    c_lib.wowml_tensor_from_data.restype = ctypes.c_int
    c_lib.wowml_tensor_zeros.argtypes = [T_PTR, ctypes.c_int, ctypes.c_int, I64_PTR]
    c_lib.wowml_tensor_zeros.restype = ctypes.c_int
    c_lib.wowml_tensor_ones.argtypes = [T_PTR, ctypes.c_int, ctypes.c_int, I64_PTR]
    c_lib.wowml_tensor_ones.restype = ctypes.c_int
    c_lib.wowml_tensor_fill.argtypes = [T_PTR, ctypes.c_double]
    c_lib.wowml_tensor_fill.restype = ctypes.c_int
    c_lib.wowml_tensor_clone.argtypes = [T_PTR, T_PTR]
    c_lib.wowml_tensor_clone.restype = ctypes.c_int
    c_lib.wowml_tensor_set_flat.argtypes = [T_PTR, ctypes.c_int64, ctypes.c_double]
    c_lib.wowml_tensor_set_flat.restype = ctypes.c_int
    c_lib.wowml_tensor_get_flat.argtypes = [T_PTR, ctypes.c_int64, DBL_PTR]
    c_lib.wowml_tensor_get_flat.restype = ctypes.c_int
    c_lib.wowml_tensor_is_contiguous.argtypes = [T_PTR]
    c_lib.wowml_tensor_is_contiguous.restype = ctypes.c_int
    c_lib.wowml_tensor_contiguous.argtypes = [T_PTR, T_PTR]
    c_lib.wowml_tensor_contiguous.restype = ctypes.c_int
    c_lib.wowml_tensor_reshape.argtypes = [T_PTR, ctypes.c_int, I64_PTR]
    c_lib.wowml_tensor_reshape.restype = ctypes.c_int
    c_lib.wowml_tensor_transpose.argtypes = [T_PTR, T_PTR, ctypes.c_int, ctypes.c_int]
    c_lib.wowml_tensor_transpose.restype = ctypes.c_int
    c_lib.wowml_tensor_slice.argtypes = [
        T_PTR, T_PTR, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64
    ]
    c_lib.wowml_tensor_slice.restype = ctypes.c_int
    c_lib.wowml_broadcast_shapes.argtypes = [
        ctypes.c_int, I64_PTR, ctypes.c_int, I64_PTR, ctypes.POINTER(ctypes.c_int), I64_PTR
    ]
    c_lib.wowml_broadcast_shapes.restype = ctypes.c_int
    c_lib.wowml_tensor_broadcast_to.argtypes = [T_PTR, T_PTR, ctypes.c_int, I64_PTR]
    c_lib.wowml_tensor_broadcast_to.restype = ctypes.c_int
    c_lib.wowml_tensor_print.argtypes = [T_PTR]
    c_lib.wowml_tensor_print.restype = None

    # ---- ops.h ----
    for binary_op in ("wowml_add", "wowml_sub", "wowml_mul", "wowml_div", "wowml_equal"):
        fn = getattr(c_lib, binary_op)
        fn.argtypes = [T_PTR, T_PTR, T_PTR]
        fn.restype = ctypes.c_int

    for scalar_op in ("wowml_add_scalar", "wowml_mul_scalar", "wowml_pow_op"):
        fn = getattr(c_lib, scalar_op)
        fn.argtypes = [T_PTR, T_PTR, ctypes.c_double]
        fn.restype = ctypes.c_int

    for unary_op in ("wowml_neg", "wowml_exp", "wowml_log", "wowml_relu", "wowml_sigmoid", "wowml_tanh_op", "wowml_sqrt"):
        fn = getattr(c_lib, unary_op)
        fn.argtypes = [T_PTR, T_PTR]
        fn.restype = ctypes.c_int

    for red_op in ("wowml_sum", "wowml_mean", "wowml_max", "wowml_min"):
        fn = getattr(c_lib, red_op)
        fn.argtypes = [T_PTR, T_PTR, ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_int

    c_lib.wowml_im2col.argtypes = [
        T_PTR, T_PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    c_lib.wowml_im2col.restype = ctypes.c_int
    c_lib.wowml_col2im.argtypes = c_lib.wowml_im2col.argtypes
    c_lib.wowml_col2im.restype = ctypes.c_int

    c_lib.wowml_fused_linear_bias_relu.argtypes = [T_PTR, T_PTR, T_PTR, T_PTR]
    c_lib.wowml_fused_linear_bias_relu.restype = ctypes.c_int
    c_lib.wowml_fused_cross_entropy.argtypes = [DBL_PTR, T_PTR, T_PTR, T_PTR]
    c_lib.wowml_fused_cross_entropy.restype = ctypes.c_int

    # ---- matmul.h ----
    for mm_op in ("wowml_matmul", "wowml_matmul_c", "wowml_matmul_blas", "wowml_bmm", "wowml_matvec"):
        fn = getattr(c_lib, mm_op)
        fn.argtypes = [T_PTR, T_PTR, T_PTR]
        fn.restype = ctypes.c_int

    c_lib.wowml_dot.argtypes = [DBL_PTR, T_PTR, T_PTR]
    c_lib.wowml_dot.restype = ctypes.c_int

    # ---- linalg.h ----
    c_lib.wowml_linalg_lu.argtypes = [T_PTR, I64_PTR, T_PTR]
    c_lib.wowml_linalg_lu.restype = ctypes.c_int
    c_lib.wowml_linalg_qr.argtypes = [T_PTR, T_PTR, T_PTR]
    c_lib.wowml_linalg_qr.restype = ctypes.c_int
    c_lib.wowml_linalg_svd.argtypes = [T_PTR, T_PTR, T_PTR, T_PTR]
    c_lib.wowml_linalg_svd.restype = ctypes.c_int
    c_lib.wowml_linalg_cholesky.argtypes = [T_PTR, T_PTR]
    c_lib.wowml_linalg_cholesky.restype = ctypes.c_int
    c_lib.wowml_linalg_eigh.argtypes = [T_PTR, T_PTR, T_PTR]
    c_lib.wowml_linalg_eigh.restype = ctypes.c_int
    c_lib.wowml_linalg_solve.argtypes = [T_PTR, T_PTR, T_PTR]
    c_lib.wowml_linalg_solve.restype = ctypes.c_int
    c_lib.wowml_linalg_inv.argtypes = [T_PTR, T_PTR]
    c_lib.wowml_linalg_inv.restype = ctypes.c_int
    c_lib.wowml_linalg_det.argtypes = [DBL_PTR, T_PTR]
    c_lib.wowml_linalg_det.restype = ctypes.c_int
    c_lib.wowml_linalg_trace.argtypes = [DBL_PTR, T_PTR]
    c_lib.wowml_linalg_trace.restype = ctypes.c_int
    c_lib.wowml_linalg_norm.argtypes = [DBL_PTR, T_PTR, ctypes.c_int]
    c_lib.wowml_linalg_norm.restype = ctypes.c_int


try:
    lib: Optional[ctypes.CDLL] = ctypes.CDLL(str(_find_library()))
    _setup_bindings(lib)
except OSError:
    lib = None
