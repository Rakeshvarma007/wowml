"""
engine.py — Reverse-mode automatic differentiation engine.
Supports dynamic DAG traversal, no_grad context, higher-order derivatives, and autograd.grad.
"""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from wowml.tensor import Tensor

_GRAD_ENABLED = True


@contextmanager
def no_grad():
    """Context manager disabling autograd graph tracking."""
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def is_grad_enabled() -> bool:
    """Return whether gradient tracking is currently active."""
    return _GRAD_ENABLED


def backward(tensor: "Tensor", gradient: Optional["Tensor"] = None, retain_graph: bool = False) -> None:
    """
    Run reverse-mode automatic differentiation starting from tensor.
    Accumulates gradients into .grad on leaf nodes.
    """
    # TODO: Implement reverse-mode autodiff stub
    raise NotImplementedError


def grad(outputs: Sequence["Tensor"], inputs: Sequence["Tensor"],
         grad_outputs: Optional[Sequence["Tensor"]] = None,
         retain_graph: bool = False, create_graph: bool = False) -> Tuple["Tensor", ...]:
    """
    Compute and return the sum of gradients of outputs with respect to inputs.
    Supports higher-order derivatives if create_graph is True.
    """
    # TODO: Implement functional autograd.grad stub
    raise NotImplementedError


def _topological_sort(root: "Tensor") -> Sequence["Tensor"]:
    """Return nodes in reverse topological order."""
    # TODO: Implement DFS topological sorting stub
    raise NotImplementedError
