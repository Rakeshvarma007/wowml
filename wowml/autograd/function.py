"""
function.py — Base class for differentiable operations.
"""
from __future__ import annotations
from typing import Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wowml.tensor import Tensor


class Function:
    """
    Base class for autograd functions.
    Subclasses implement forward() and backward().
    """

    def __init__(self):
        self.inputs: Tuple["Tensor", ...] = ()
        self.saved_tensors: Tuple[Any, ...] = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def forward(self, *args) -> "Tensor":
        raise NotImplementedError

    def backward(self, grad_output: "Tensor") -> Tuple["Tensor", ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args) -> "Tensor":
        fn = cls()
        fn.inputs = args
        result = fn.forward(*args)
        result._grad_fn = fn
        return result
