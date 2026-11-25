"""
tensor.py — Python-facing Tensor class with full operator overloads, views, and autograd hooks.
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union


class Tensor:
    """
    Multi-dimensional array backed by C memory with automatic differentiation.
    """

    def __init__(self, data, dtype: str = "float32", requires_grad: bool = False):
        self.data = data
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._grad_fn = None
        self._c_tensor = None

    @property
    def shape(self) -> Tuple[int, ...]: raise NotImplementedError
    @property
    def ndim(self) -> int: raise NotImplementedError
    @property
    def numel(self) -> int: raise NotImplementedError
    @property
    def strides(self) -> Tuple[int, ...]: raise NotImplementedError
    @property
    def is_contiguous(self) -> bool: raise NotImplementedError

    @staticmethod
    def zeros(*shape, dtype: str = "float32", requires_grad: bool = False) -> Tensor: raise NotImplementedError
    @staticmethod
    def ones(*shape, dtype: str = "float32", requires_grad: bool = False) -> Tensor: raise NotImplementedError
    @staticmethod
    def randn(*shape, dtype: str = "float32", requires_grad: bool = False) -> Tensor: raise NotImplementedError

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor: raise NotImplementedError
    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor: raise NotImplementedError
    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor: raise NotImplementedError
    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor: raise NotImplementedError
    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor: raise NotImplementedError
    def __rmul__(self, other: Union[Tensor, float, int]) -> Tensor: raise NotImplementedError
    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor: raise NotImplementedError
    def __rtruediv__(self, other: Union[Tensor, float, int]) -> Tensor: raise NotImplementedError
    def __matmul__(self, other: Tensor) -> Tensor: raise NotImplementedError
    def __neg__(self) -> Tensor: raise NotImplementedError
    def __pow__(self, exponent: float) -> Tensor: raise NotImplementedError

    def __getitem__(self, item) -> Tensor: raise NotImplementedError
    def __setitem__(self, key, value) -> None: raise NotImplementedError

    def backward(self, gradient: Optional[Tensor] = None, retain_graph: bool = False) -> None: raise NotImplementedError
    def zero_grad(self) -> None: raise NotImplementedError
    def detach(self) -> Tensor: raise NotImplementedError

    def reshape(self, *shape: int) -> Tensor: raise NotImplementedError
    def transpose(self, dim0: int, dim1: int) -> Tensor: raise NotImplementedError
    @property
    def T(self) -> Tensor: raise NotImplementedError
    def contiguous(self) -> Tensor: raise NotImplementedError

    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor: raise NotImplementedError
    def mean(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> Tensor: raise NotImplementedError

    def to(self, dtype: str) -> Tensor: raise NotImplementedError
    def numpy(self): raise NotImplementedError
    def tolist(self): raise NotImplementedError

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape if hasattr(self, '_shape') else '...'}, dtype='{self.dtype}')"
