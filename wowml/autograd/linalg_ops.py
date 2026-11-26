"""
linalg_ops.py — Differentiable linear algebra operations (Solve, Cholesky).
"""
from __future__ import annotations
from wowml.autograd.function import Function
from wowml.linalg import solve as _solve, cholesky as _cholesky, inv as _inv
from wowml.tensor import Tensor


class Solve(Function):
    """Differentiable linear system solve: A @ X = B."""
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        x = _solve(a, b)
        self.save_for_backward(a, x)
        return x

    def backward(self, grad_output: Tensor):
        a, x = self.saved_tensors
        # dL/dB = A^-T @ grad_output, dL/dA = -dL/dB @ X^T
        grad_b = _solve(a.T, grad_output)
        if grad_b.ndim == 1:
            import numpy as np
            gb = grad_b.numpy()[:, None]
            xt = x.numpy()[None, :]
            grad_a = Tensor(-(gb @ xt), dtype=a.dtype)
        else:
            grad_a = -(grad_b @ x.T)
        return grad_a, grad_b


class Cholesky(Function):
    """Differentiable Cholesky decomposition: A = L @ L^T."""
    def forward(self, a: Tensor) -> Tensor:
        l = _cholesky(a)
        self.save_for_backward(l)
        return l

    def backward(self, grad_output: Tensor):
        (l,) = self.saved_tensors
        import numpy as np
        L = l.numpy()
        dL = grad_output.numpy()
        # Symmetrize phi(L^T dL)
        M = L.T @ dL
        P = np.tril(M)
        np.fill_diagonal(P, 0.5 * np.diag(P))
        Linv = np.linalg.inv(L)
        S = Linv.T @ (P + P.T) @ Linv
        grad_a = 0.5 * (S + S.T)
        return (Tensor(grad_a.astype(l.dtype), dtype=l.dtype),)
