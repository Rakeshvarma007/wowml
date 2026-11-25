"""
grad_ops.py — All 17 differentiable operations with broadcast unrolling stubs.
"""
from __future__ import annotations
from wowml.autograd.function import Function


def _unbroadcast(grad, target_shape):
    """Sum out dimensions added by broadcasting."""
    # TODO: Implement broadcast reduction stub
    raise NotImplementedError


class Add(Function):
    def forward(self, a, b): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Sub(Function):
    def forward(self, a, b): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Mul(Function):
    def forward(self, a, b): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Div(Function):
    def forward(self, a, b): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Neg(Function):
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class MatMul(Function):
    def forward(self, a, b): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Pow(Function):
    def forward(self, x, exponent): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Sqrt(Function):
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Exp(Function):
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Log(Function):
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class ReLU(Function):
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Sigmoid(Function):
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Tanh(Function):
    def forward(self, x): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Sum(Function):
    def forward(self, x, axis=None, keepdims=False): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Mean(Function):
    def forward(self, x, axis=None, keepdims=False): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Reshape(Function):
    def forward(self, x, shape): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError

class Transpose(Function):
    def forward(self, x, dim0, dim1): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError
