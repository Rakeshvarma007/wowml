"""
_C — Low-level ctypes bindings and zero-copy interop with the wowml_c shared library.
"""
from wowml._C._bindings import (  # noqa: F401
    lib,
    DType,
    Status,
    CWowmlTensor,
    CWowmlArena,
    DTYPE_FROM_STR,
    DTYPE_TO_STR,
    DTYPE_TO_CTYPE,
    check_status,
    require_lib,
    make_c_shape,
    c_tensor_from_numpy,
    numpy_from_c_tensor,
)
