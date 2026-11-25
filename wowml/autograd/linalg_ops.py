"""
linalg_ops.py — Differentiable linear algebra operations for the autograd engine.
"""
from __future__ import annotations
from wowml.autograd.function import Function


class Solve(Function):
    """Differentiable linear system solve: A @ X = B."""
    def forward(self, a, b):
        # TODO: Implement forward solve stub
        raise NotImplementedError

    def backward(self, grad_output):
        # TODO: Implement dL/dA = - (A^-T @ grad_output) @ X^T, dL/dB = A^-T @ grad_output
        raise NotImplementedError


class Cholesky(Function):
    """Differentiable Cholesky decomposition: A = L @ L^T."""
    def forward(self, a):
        # TODO: Implement forward cholesky stub
        raise NotImplementedError

    def backward(self, grad_output):
        # TODO: Implement backward gradient for Cholesky
        raise NotImplementedError


class SVD(Function):
    """Differentiable SVD: A = U @ diag(S) @ V^T."""
    def forward(self, a):
        # TODO: Implement forward SVD stub
        raise NotImplementedError

    def backward(self, grad_u, grad_s, grad_vt):
        # TODO: Implement backward gradient for SVD
        raise NotImplementedError
