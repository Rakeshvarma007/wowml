"""
linear.py — Fully-connected (dense) layer.
"""
import math
from wowml.nn.module import Module
from wowml.nn.parameter import Parameter
from wowml.tensor import Tensor


class Linear(Module):
    """y = x @ W^T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: str = "float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        bound = 1.0 / math.sqrt(in_features)
        import numpy as np
        w_arr = np.random.uniform(-bound, bound, (out_features, in_features)).astype(dtype)
        self.weight = Parameter(w_arr, dtype=dtype)
        if bias:
            b_arr = np.random.uniform(-bound, bound, (out_features,)).astype(dtype)
            self.bias = Parameter(b_arr, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
