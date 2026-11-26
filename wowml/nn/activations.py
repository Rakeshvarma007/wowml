"""
activations.py — Activation function modules wired to autograd.
"""
import math
from wowml.autograd import grad_ops
from wowml.nn.module import Module
from wowml.tensor import Tensor


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return grad_ops.ReLU.apply(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return grad_ops.Sigmoid.apply(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return grad_ops.Tanh.apply(x)


class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        x_max = Tensor(x.numpy().max(axis=self.dim, keepdims=True).astype(x.dtype), dtype=x.dtype)
        ex = grad_ops.Exp.apply(x - x_max)
        return ex / ex.sum(axis=self.dim, keepdims=True)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        pos = grad_ops.ReLU.apply(x)
        neg = grad_ops.ReLU.apply(-x) * (-self.negative_slope)
        return pos + neg


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        c = math.sqrt(2.0 / math.pi)
        inner = (x + 0.044715 * (x ** 3.0)) * c
        return 0.5 * x * (1.0 + grad_ops.Tanh.apply(inner))


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * grad_ops.Sigmoid.apply(x)
