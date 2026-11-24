"""
grad_ops.py — Concrete autograd Function implementations for all ops.
"""
from __future__ import annotations
from wowml.autograd.function import Function


class Add(Function):
    """z = a + b"""
    def forward(self, a, b):        raise NotImplementedError
    def backward(self, grad_output): return grad_output, grad_output


class Sub(Function):
    """z = a - b"""
    def forward(self, a, b):        raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Mul(Function):
    """z = a * b"""
    def forward(self, a, b):
        self.save_for_backward(a, b)
        raise NotImplementedError
    def backward(self, grad_output):
        a, b = self.saved_tensors
        raise NotImplementedError


class Div(Function):
    """z = a / b"""
    def forward(self, a, b):
        self.save_for_backward(a, b)
        raise NotImplementedError
    def backward(self, grad_output):
        a, b = self.saved_tensors
        raise NotImplementedError


class Neg(Function):
    def forward(self, x):           raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class MatMul(Function):
    """z = A @ B"""
    def forward(self, a, b):
        self.save_for_backward(a, b)
        raise NotImplementedError
    def backward(self, grad_output):
        a, b = self.saved_tensors
        raise NotImplementedError


class Pow(Function):
    def forward(self, x, exponent):
        self.save_for_backward(x, exponent)
        raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Sqrt(Function):
    def forward(self, x):           raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Exp(Function):
    def forward(self, x):           raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Log(Function):
    def forward(self, x):           raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class ReLU(Function):
    """z = max(0, x)"""
    def forward(self, x):
        self.save_for_backward(x)
        raise NotImplementedError
    def backward(self, grad_output):
        (x,) = self.saved_tensors
        raise NotImplementedError


class Sigmoid(Function):
    def forward(self, x):           raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Tanh(Function):
    def forward(self, x):           raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Sum(Function):
    def forward(self, x, axis=None): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Mean(Function):
    def forward(self, x, axis=None): raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Reshape(Function):
    def forward(self, x, shape):    raise NotImplementedError
    def backward(self, grad_output): raise NotImplementedError


class Transpose(Function):
    def forward(self, x, dim0, dim1): raise NotImplementedError
    def backward(self, grad_output):  raise NotImplementedError
