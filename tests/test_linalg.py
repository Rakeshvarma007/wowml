"""Tests for the linear algebra module against NumPy."""
import numpy as np
import pytest
import wowml
from wowml import Tensor


class TestDecompositions:
    def test_qr(self):
        a_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        q, r = wowml.linalg.qr(Tensor(a_np, dtype="float64"))
        np.testing.assert_allclose(q.numpy() @ r.numpy(), a_np, atol=1e-6)
        np.testing.assert_allclose(q.numpy().T @ q.numpy(), np.eye(2), atol=1e-6)

    def test_cholesky(self):
        m = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float64)
        l = wowml.linalg.cholesky(Tensor(m, dtype="float64"))
        np.testing.assert_allclose(l.numpy() @ l.numpy().T, m, atol=1e-6)

    def test_svd(self):
        a_np = np.array([[3.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        u, s, vt = wowml.linalg.svd(Tensor(a_np, dtype="float64"))
        rec = u.numpy() @ np.diag(s.numpy()) @ vt.numpy()
        np.testing.assert_allclose(rec, a_np, atol=1e-5)

    def test_eigh(self):
        a_np = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        w, v = wowml.linalg.eigh(Tensor(a_np, dtype="float64"))
        np.testing.assert_allclose(v.numpy() @ np.diag(w.numpy()) @ v.numpy().T, a_np, atol=1e-6)


class TestSolvers:
    def test_solve_and_inv(self):
        a_np = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        b_np = np.array([9.0, 8.0], dtype=np.float64)
        x = wowml.linalg.solve(Tensor(a_np, dtype="float64"), Tensor(b_np, dtype="float64"))
        np.testing.assert_allclose(x.numpy(), np.linalg.solve(a_np, b_np), atol=1e-6)
        inv_a = wowml.linalg.inv(Tensor(a_np, dtype="float64"))
        np.testing.assert_allclose(inv_a.numpy(), np.linalg.inv(a_np), atol=1e-6)


class TestNorms:
    def test_det_trace_norm(self):
        a_np = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float64)
        a = Tensor(a_np, dtype="float64")
        assert abs(wowml.linalg.det(a) - 8.0) < 1e-6
        assert abs(wowml.linalg.trace(a) - 7.0) < 1e-6
        assert abs(wowml.linalg.norm(a, ord=2) - np.linalg.norm(a_np)) < 1e-6
