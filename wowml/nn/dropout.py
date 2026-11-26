"""
dropout.py — Dropout regularization layer.
"""
from __future__ import annotations
import numpy as np
from wowml.nn.module import Module
from wowml.tensor import Tensor


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        scale = 1.0 / (1.0 - self.p)
        mask = (np.random.rand(*x.shape) >= self.p).astype(x.dtype) * scale
        return x * Tensor(mask, dtype=x.dtype)
