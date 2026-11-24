"""
solve.py — Linear system solvers.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wowml.tensor import Tensor


def solve(a: "Tensor", b: "Tensor") -> "Tensor":
    """Solve Ax = b for x."""
    raise NotImplementedError

def inv(a: "Tensor") -> "Tensor":
    """Compute the inverse of a square matrix."""
    raise NotImplementedError
