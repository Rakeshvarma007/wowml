"""
linear.py — Fully-connected (dense) layer.
"""
from wowml.nn.module import Module
from wowml.nn.parameter import Parameter
from wowml.tensor import Tensor


class Linear(Module):
    """y = x @ W^T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO: proper random init
        self.weight = Parameter(None)
        self.bias = Parameter(None) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
