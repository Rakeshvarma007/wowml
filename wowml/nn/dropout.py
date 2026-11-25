"""
dropout.py — Dropout regularization layer.
"""
from wowml.nn.module import Module
from wowml.tensor import Tensor


class Dropout(Module):
    """Randomly zeroes elements with probability p during training."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
