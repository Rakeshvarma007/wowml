"""Tests for the Python Tensor class backed by the C kernel."""
import numpy as np
import pytest
import wowml
from wowml import Tensor


class TestTensorCreation:
    def test_zeros_ones(self):
        z = Tensor.zeros(2, 3)
        o = Tensor.ones(2, 3)
        assert z.shape == (2, 3)
        assert o.shape == (2, 3)
        np.testing.assert_allclose(z.numpy(), np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_allclose(o.numpy(), np.ones((2, 3), dtype=np.float32))

    def test_from_numpy_zero_copy(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor.from_numpy(arr)
        assert t.shape == (2, 2)
        np.testing.assert_allclose(t.numpy(), arr)


class TestTensorArithmetic:
    def test_add_sub_mul_div(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[2.0, 2.0], [2.0, 2.0]])
        np.testing.assert_allclose((a + b).numpy(), [[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose((a - b).numpy(), [[-1.0, 0.0], [1.0, 2.0]])
        np.testing.assert_allclose((a * b).numpy(), [[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose((a / b).numpy(), [[0.5, 1.0], [1.5, 2.0]])

    def test_matmul(self):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = a @ b
        np.testing.assert_allclose(c.numpy(), [[22.0, 28.0], [49.0, 64.0]])


class TestTensorReshape:
    def test_reshape_transpose(self):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = a.reshape(3, 2)
        assert r.shape == (3, 2)
        t = a.T
        assert t.shape == (3, 2)
        np.testing.assert_allclose(t.contiguous().numpy(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
