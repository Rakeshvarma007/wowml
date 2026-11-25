"""
pooling.py — 2D pooling layers (MaxPool2d and AvgPool2d).
"""
from wowml.nn.module import Module
from wowml.tensor import Tensor


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class AvgPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
