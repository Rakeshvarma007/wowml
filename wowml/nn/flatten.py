"""
flatten.py — Flatten layer.
"""
from wowml.nn.module import Module
from wowml.tensor import Tensor


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
