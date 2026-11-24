"""
norms.py — Norms, determinant, trace.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wowml.tensor import Tensor


def norm(a: "Tensor", ord: int = 2) -> float:
    raise NotImplementedError

def det(a: "Tensor") -> float:
    raise NotImplementedError

def trace(a: "Tensor") -> float:
    raise NotImplementedError
