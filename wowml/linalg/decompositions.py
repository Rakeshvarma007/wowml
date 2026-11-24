"""
decompositions.py — Matrix decomposition wrappers.
"""
from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from wowml.tensor import Tensor


def lu(a: "Tensor") -> Tuple["Tensor", "Tensor"]:
    """LU decomposition with partial pivoting."""
    raise NotImplementedError

def qr(a: "Tensor") -> Tuple["Tensor", "Tensor"]:
    """QR decomposition."""
    raise NotImplementedError

def svd(a: "Tensor") -> Tuple["Tensor", "Tensor", "Tensor"]:
    """Singular Value Decomposition."""
    raise NotImplementedError

def cholesky(a: "Tensor") -> "Tensor":
    """Cholesky decomposition (lower triangular)."""
    raise NotImplementedError

def eigh(a: "Tensor") -> Tuple["Tensor", "Tensor"]:
    """Eigenvalue decomposition for symmetric matrices."""
    raise NotImplementedError
