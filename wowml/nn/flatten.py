"""
flatten.py — Flatten layer.
"""
import math
from wowml.nn.module import Module
from wowml.tensor import Tensor


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        s = self.start_dim if self.start_dim >= 0 else self.start_dim + x.ndim
        e = self.end_dim if self.end_dim >= 0 else self.end_dim + x.ndim
        flat_dim = math.prod(x.shape[s:e + 1])
        new_shape = x.shape[:s] + (flat_dim,) + x.shape[e + 1:]
        return x.reshape(*new_shape)
