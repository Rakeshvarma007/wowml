"""
tensor.py — The Python-side Tensor class.

Wraps the C kernel through ctypes and hooks into the autograd engine.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math


class Tensor:
    """
    Multi-dimensional array with optional autograd support.

    Attributes:
        data: Raw numerical data (will be C-backed).
        shape: Tuple of dimension sizes.
        dtype: Data type string.
        requires_grad: Whether this tensor participates in autograd.
        grad: Accumulated gradient (Tensor or None).
        _grad_fn: The autograd Function that created this tensor.
    """

    def __init__(self, data, dtype: str = "float32", requires_grad: bool = False):
        # TODO: Replace Python storage with C-backed buffer
        self.data = data
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._grad_fn = None

    @property
    def shape(self) -> Tuple[int, ...]:
        # TODO: derive from C tensor
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    # ---- Factory methods ----

    @staticmethod
    def zeros(*shape, dtype="float32", requires_grad=False) -> "Tensor":
        raise NotImplementedError

    @staticmethod
    def ones(*shape, dtype="float32", requires_grad=False) -> "Tensor":
        raise NotImplementedError

    @staticmethod
    def randn(*shape, dtype="float32", requires_grad=False) -> "Tensor":
        raise NotImplementedError

    # ---- Arithmetic ----

    def __add__(self, other):       raise NotImplementedError
    def __mul__(self, other):       raise NotImplementedError
    def __matmul__(self, other):    raise NotImplementedError
    def __sub__(self, other):       raise NotImplementedError
    def __truediv__(self, other):   raise NotImplementedError
    def __neg__(self):              raise NotImplementedError

    # ---- Autograd ----

    def backward(self, gradient: Optional["Tensor"] = None):
        raise NotImplementedError

    def zero_grad(self):
        self.grad = None

    # ---- Shape ops ----

    def reshape(self, *shape) -> "Tensor":          raise NotImplementedError
    def transpose(self, dim0: int, dim1: int):      raise NotImplementedError

    @property
    def T(self) -> "Tensor":                        raise NotImplementedError

    # ---- Reductions ----

    def sum(self, axis=None) -> "Tensor":           raise NotImplementedError
    def mean(self, axis=None) -> "Tensor":          raise NotImplementedError

    # ---- Misc ----

    def detach(self) -> "Tensor":                   raise NotImplementedError

    def __repr__(self):
        g = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor(data=..., dtype={self.dtype}{g})"
