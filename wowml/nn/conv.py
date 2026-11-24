"""
conv.py — Convolution layers.
"""
from wowml.nn.module import Module
from wowml.nn.parameter import Parameter
from wowml.tensor import Tensor


class Conv2d(Module):
    """2D convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Parameter(None)
        self.bias_param = Parameter(None) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")
