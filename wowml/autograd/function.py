"""
function.py — Base class for differentiable operations.
Handles autograd graph recording when grad tracking is enabled.
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
        self.inputs: Tuple[Any, ...] = ()
        self.saved_tensors: Tuple[Any, ...] = ()

    def save_for_backward(self, *tensors: Any) -> None:
        self.saved_tensors = tensors

    def forward(self, *args, **kwargs) -> "Tensor":
        raise NotImplementedError

    def backward(self, grad_output: "Tensor") -> Tuple[Any, ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs) -> "Tensor":
        from wowml.autograd.engine import is_grad_enabled
        from wowml.tensor import Tensor

        fn = cls()
        fn.inputs = args
        result = fn.forward(*args, **kwargs)

        needs_grad = is_grad_enabled() and any(
            isinstance(a, Tensor) and a.requires_grad for a in args
        )
        if isinstance(result, Tensor) and needs_grad:
            result.requires_grad = True
            result._grad_fn = fn
        return result
