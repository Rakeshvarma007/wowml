"""
activations.py — Activation function modules.
"""
from wowml.nn.module import Module
from wowml.tensor import Tensor


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError

class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError

class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError

class GELU(Module):
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError
