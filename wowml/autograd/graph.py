"""
graph.py — Computation graph utilities for inspection and debugging.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wowml.tensor import Tensor


def trace_graph(tensor: "Tensor") -> str:
    """Return a human-readable string of the computation graph."""
    raise NotImplementedError


def count_parameters(tensor: "Tensor") -> int:
    """Count total leaf parameters in the graph."""
    raise NotImplementedError
