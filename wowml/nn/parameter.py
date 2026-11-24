"""
parameter.py — Parameter class (a Tensor that is a learnable weight).
"""
from wowml.tensor import Tensor


class Parameter(Tensor):
    """A Tensor subclass registered as a module parameter. Always requires_grad=True."""

    def __init__(self, data, dtype: str = "float32"):
        super().__init__(data, dtype=dtype, requires_grad=True)

    def __repr__(self):
        return f"Parameter(dtype={self.dtype})"
