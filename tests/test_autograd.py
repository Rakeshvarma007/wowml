"""Tests for the autograd engine including finite-difference numerical gradient checks."""
import numpy as np
import pytest
import wowml
from wowml import Tensor
from wowml.autograd import grad, no_grad


class TestAutogradBasic:
    def test_add_mul_chain(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = Tensor([4.0], requires_grad=True)
        y = a * b + c
        y.backward()
        np.testing.assert_allclose(a.grad.numpy(), [3.0])
        np.testing.assert_allclose(b.grad.numpy(), [2.0])
        np.testing.assert_allclose(c.grad.numpy(), [1.0])

    def test_no_grad_context(self):
        a = Tensor([2.0], requires_grad=True)
        with no_grad():
            b = a * 3.0
        assert not b.requires_grad
        assert b._grad_fn is None

    def test_functional_grad(self):
        x = Tensor([3.0], requires_grad=True)
        y = x * x * x
        (dx,) = grad(y, x)
        np.testing.assert_allclose(dx.numpy(), [27.0])


class TestAutogradMatmul:
    def test_matmul_backward(self):
        a_np = np.random.randn(3, 4).astype(np.float64)
        b_np = np.random.randn(4, 2).astype(np.float64)
        a = Tensor(a_np.copy(), dtype="float64", requires_grad=True)
        b = Tensor(b_np.copy(), dtype="float64", requires_grad=True)
        loss = (a @ b).sum()
        loss.backward()
        np.testing.assert_allclose(a.grad.numpy(), np.ones((3, 2)) @ b_np.T, atol=1e-6)
        np.testing.assert_allclose(b.grad.numpy(), a_np.T @ np.ones((3, 2)), atol=1e-6)


class TestAutogradNumericalGradient:
    def test_numerical_vs_analytical(self):
        from wowml.autograd import grad_ops
        x_np = np.array([[0.5, -1.2], [1.7, 0.3]], dtype=np.float64)
        x = Tensor(x_np.copy(), dtype="float64", requires_grad=True)
        y = (grad_ops.Sigmoid.apply(x) ** 2.0).sum()
        y.backward()
        analytic = x.grad.numpy()

        eps = 1e-6
        numeric = np.zeros_like(x_np)
        for i in range(x_np.shape[0]):
            for j in range(x_np.shape[1]):
                xp = x_np.copy(); xp[i, j] += eps
                xm = x_np.copy(); xm[i, j] -= eps
                fp = (1.0 / (1.0 + np.exp(-xp)) ** 1.0)
                fp = (fp ** 2.0).sum()
                fm = (1.0 / (1.0 + np.exp(-xm)) ** 1.0)
                fm = (fm ** 2.0).sum()
                numeric[i, j] = (fp - fm) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, atol=1e-5)
