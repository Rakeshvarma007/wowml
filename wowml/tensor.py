"""
tensor.py — Python-facing Tensor class wired to the C kernel and autograd engine.
"""
from __future__ import annotations
import ctypes
import math
from typing import Any, Optional, Sequence, Tuple, Union

from wowml._C._bindings import (
    CWowmlTensor,
    DType,
    DTYPE_FROM_STR,
    DTYPE_TO_CTYPE,
    DTYPE_TO_STR,
    c_tensor_from_numpy,
    check_status,
    lib,
    make_c_shape,
    numpy_from_c_tensor,
    require_lib,
)


def _infer_shape_and_flatten(data: Any) -> Tuple[Tuple[int, ...], list]:
    """Infer shape from nested Python sequences and return (shape, flat_list)."""
    if isinstance(data, (int, float, bool)):
        return (1,), [float(data)]
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Unsupported data type for Tensor: {type(data)}")
    if len(data) == 0:
        return (0,), []
    if not isinstance(data[0], (list, tuple)):
        return (len(data),), list(data)
    sub_shapes = []
    flat = []
    for item in data:
        s, f = _infer_shape_and_flatten(item)
        sub_shapes.append(s)
        flat.extend(f)
    first = sub_shapes[0]
    for s in sub_shapes[1:]:
        if s != first:
            raise ValueError("Ragged nested sequences cannot form a Tensor.")
    return (len(data),) + first, flat


class Tensor:
    """
    Multi-dimensional array backed by C memory with automatic differentiation.
    """

    def __init__(
        self,
        data: Any = None,
        dtype: str = "float32",
        requires_grad: bool = False,
        _c_tensor: Optional[CWowmlTensor] = None,
    ):
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._grad_fn = None
        self._c_tensor: Optional[CWowmlTensor] = None
        self._np_ref = None
        self._shape_fallback: Optional[Tuple[int, ...]] = None

        if _c_tensor is not None:
            self._c_tensor = _c_tensor
            self.dtype = DTYPE_TO_STR.get(DType(_c_tensor.dtype), dtype)
            return

        if data is None:
            return

        if dtype not in DTYPE_FROM_STR:
            raise ValueError(f"Unsupported dtype '{dtype}'. Expected one of {list(DTYPE_FROM_STR)}.")

        # Fast zero-copy path for NumPy arrays
        if hasattr(data, "__array_interface__"):
            self._np_ref = data
            self._shape_fallback = tuple(data.shape)
            self.dtype = str(data.dtype)
            if lib is not None:
                self._c_tensor = c_tensor_from_numpy(data)
                self.dtype = DTYPE_TO_STR[DType(self._c_tensor.dtype)]
            return

        # Python sequence / scalar path
        shape, flat = _infer_shape_and_flatten(data)
        self._shape_fallback = shape
        if lib is not None:
            c_dtype = DTYPE_FROM_STR[dtype]
            t = CWowmlTensor()
            ndim, c_shape = make_c_shape(shape)
            check_status(lib.wowml_tensor_alloc(ctypes.byref(t), int(c_dtype), ndim, c_shape))
            c_type = DTYPE_TO_CTYPE[c_dtype]
            buf = (c_type * len(flat))(*flat)
            ctypes.memmove(t.data, buf, len(flat) * ctypes.sizeof(c_type))
            self._c_tensor = t

    def __del__(self):
        if getattr(self, "_c_tensor", None) is not None and lib is not None:
            try:
                lib.wowml_tensor_free(ctypes.byref(self._c_tensor))
            except Exception:
                pass
            self._c_tensor = None

    @property
    def c_ptr(self):
        """Return ctypes pointer to underlying wowml_tensor_t."""
        if self._c_tensor is None:
            raise RuntimeError("Tensor has no allocated C buffer.")
        return ctypes.byref(self._c_tensor)

    @property
    def shape(self) -> Tuple[int, ...]:
        if self._c_tensor is not None:
            return tuple(self._c_tensor.shape[i] for i in range(self._c_tensor.ndim))
        if self._shape_fallback is not None:
            return self._shape_fallback
        return ()

    @property
    def ndim(self) -> int:
        return self._c_tensor.ndim if self._c_tensor is not None else len(self.shape)

    @property
    def numel(self) -> int:
        return self._c_tensor.numel if self._c_tensor is not None else math.prod(self.shape)

    @property
    def strides(self) -> Tuple[int, ...]:
        if self._c_tensor is not None:
            return tuple(self._c_tensor.strides[i] for i in range(self._c_tensor.ndim))
        raise RuntimeError("Strides require an allocated C tensor.")

    @property
    def is_contiguous(self) -> bool:
        if self._c_tensor is not None and lib is not None:
            return bool(lib.wowml_tensor_is_contiguous(self.c_ptr))
        return True

    # ---- Factory methods ----

    @staticmethod
    def from_numpy(arr, requires_grad: bool = False) -> Tensor:
        """Create a zero-copy Tensor wrapping a contiguous NumPy array."""
        return Tensor(arr, requires_grad=requires_grad)

    @staticmethod
    def zeros(*shape: int, dtype: str = "float32", requires_grad: bool = False) -> Tensor:
        c_lib = require_lib()
        c_dtype = DTYPE_FROM_STR[dtype]
        t = CWowmlTensor()
        ndim, c_shape = make_c_shape(shape)
        check_status(c_lib.wowml_tensor_zeros(ctypes.byref(t), int(c_dtype), ndim, c_shape))
        return Tensor(dtype=dtype, requires_grad=requires_grad, _c_tensor=t)

    @staticmethod
    def ones(*shape: int, dtype: str = "float32", requires_grad: bool = False) -> Tensor:
        c_lib = require_lib()
        c_dtype = DTYPE_FROM_STR[dtype]
        t = CWowmlTensor()
        ndim, c_shape = make_c_shape(shape)
        check_status(c_lib.wowml_tensor_ones(ctypes.byref(t), int(c_dtype), ndim, c_shape))
        return Tensor(dtype=dtype, requires_grad=requires_grad, _c_tensor=t)

    @staticmethod
    def randn(*shape: int, dtype: str = "float32", requires_grad: bool = False) -> Tensor:
        import numpy as np
        np_dtype = "float32" if dtype == "float32" else "float64"
        arr = np.random.randn(*shape).astype(np_dtype)
        t = Tensor(arr, dtype=dtype, requires_grad=requires_grad)
        return t

    # ---- Arithmetic & Autograd Operator Wiring ----

    @staticmethod
    def _ensure_tensor(other: Union[Tensor, float, int], dtype: str) -> Tensor:
        return other if isinstance(other, Tensor) else Tensor([other], dtype=dtype)

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        from wowml.autograd.grad_ops import Add
        return Add.apply(self, self._ensure_tensor(other, self.dtype))

    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor:
        from wowml.autograd.grad_ops import Add
        return Add.apply(self._ensure_tensor(other, self.dtype), self)

    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        from wowml.autograd.grad_ops import Sub
        return Sub.apply(self, self._ensure_tensor(other, self.dtype))

    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor:
        from wowml.autograd.grad_ops import Sub
        return Sub.apply(self._ensure_tensor(other, self.dtype), self)

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        from wowml.autograd.grad_ops import Mul
        return Mul.apply(self, self._ensure_tensor(other, self.dtype))

    def __rmul__(self, other: Union[Tensor, float, int]) -> Tensor:
        from wowml.autograd.grad_ops import Mul
        return Mul.apply(self._ensure_tensor(other, self.dtype), self)

    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        from wowml.autograd.grad_ops import Div
        return Div.apply(self, self._ensure_tensor(other, self.dtype))

    def __rtruediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        from wowml.autograd.grad_ops import Div
        return Div.apply(self._ensure_tensor(other, self.dtype), self)

    def __matmul__(self, other: Tensor) -> Tensor:
        from wowml.autograd.grad_ops import MatMul
        return MatMul.apply(self, other)

    def __neg__(self) -> Tensor:
        from wowml.autograd.grad_ops import Neg
        return Neg.apply(self)

    def __pow__(self, exponent: float) -> Tensor:
        from wowml.autograd.grad_ops import Pow
        return Pow.apply(self, exponent)

    def __getitem__(self, item) -> Tensor:
        c_lib = require_lib()
        if not isinstance(item, tuple):
            item = (item,)
        # TODO: Wire general multi-axis slicing to wowml_tensor_slice
        raise NotImplementedError("Slicing wiring pending wowml_tensor_slice implementation")

    # ---- Autograd ----

    def backward(self, gradient: Optional[Tensor] = None, retain_graph: bool = False) -> None:
        from wowml.autograd.engine import backward
        backward(self, gradient=gradient, retain_graph=retain_graph)

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> Tensor:
        """Return a new Tensor sharing the same underlying C data without gradient history."""
        if self._c_tensor is None:
            return Tensor(None, dtype=self.dtype, requires_grad=False)
        c_lib = require_lib()
        view = CWowmlTensor()
        ndim, c_shape = make_c_shape(self.shape)
        _, c_strides = make_c_shape(self.strides)
        check_status(c_lib.wowml_tensor_from_data(
            ctypes.byref(view),
            self._c_tensor.data,
            self._c_tensor.dtype,
            ndim,
            c_shape,
            c_strides,
            0,
        ))
        out = Tensor(dtype=self.dtype, requires_grad=False, _c_tensor=view)
        out._np_ref = self  # Keep owner alive
        return out

    # ---- Shape & Reductions ----

    def reshape(self, *shape: int) -> Tensor:
        from wowml.autograd.grad_ops import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape)

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        from wowml.autograd.grad_ops import Transpose
        return Transpose.apply(self, dim0, dim1)

    @property
    def T(self) -> Tensor:
        if self.ndim != 2:
            raise ValueError(f"Tensor.T requires 2D tensor, got {self.ndim}D.")
        return self.transpose(0, 1)

    def contiguous(self) -> Tensor:
        if self.is_contiguous:
            return self
        c_lib = require_lib()
        dst = CWowmlTensor()
        check_status(c_lib.wowml_tensor_contiguous(ctypes.byref(dst), self.c_ptr))
        return Tensor(dtype=self.dtype, requires_grad=self.requires_grad, _c_tensor=dst)

    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor:
        from wowml.autograd.grad_ops import Sum
        return Sum.apply(self, axis, keepdims)

    def mean(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor:
        from wowml.autograd.grad_ops import Mean
        return Mean.apply(self, axis, keepdims)

    # ---- Conversions ----

    def numpy(self):
        if self._c_tensor is None:
            raise RuntimeError("Tensor has no allocated C buffer.")
        return numpy_from_c_tensor(self._c_tensor)

    def tolist(self):
        return self.numpy().tolist()

    def __repr__(self) -> str:
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={self.shape}, dtype='{self.dtype}'{grad_str})"
