"""
engine.py — The core autograd engine: reverse-mode autodiff via topological traversal.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wowml.tensor import Tensor


def backward(tensor: "Tensor", gradient=None):
    """
    Compute gradients via reverse-mode autodiff.

    1. Build topological ordering of the computation graph.
    2. Seed the output gradient (default: ones).
    3. Walk the graph in reverse, calling each Function's backward().
    4. Accumulate gradients on leaf tensors.
    """
    # TODO: implement
    raise NotImplementedError("Autograd engine not yet implemented")


def _topological_sort(tensor: "Tensor") -> list:
    """Return nodes in reverse topological order."""
    visited = set()
    order = []

    def _visit(t):
        if id(t) in visited:
            return
        visited.add(id(t))
        if t._grad_fn is not None:
            for parent in t._grad_fn.inputs:
                _visit(parent)
        order.append(t)

    _visit(tensor)
    return list(reversed(order))
