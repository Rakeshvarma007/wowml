"""Tests for neural network modules, activations, and losses."""
import numpy as np
import pytest
import wowml.nn as nn
from wowml import Tensor


class TestNN:
    def test_sequential_mlp_backward(self):
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        x = Tensor(np.random.randn(5, 4).astype(np.float32))
        y = Tensor(np.random.randn(5, 2).astype(np.float32))
        loss = nn.MSELoss()(model(x), y)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert p.grad.shape == p.shape

    def test_conv2d_and_pool(self):
        conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        pool = nn.MaxPool2d(2)
        x = Tensor(np.random.randn(2, 1, 8, 8).astype(np.float32))
        out = pool(conv(x))
        assert out.shape == (2, 2, 4, 4)

    def test_cross_entropy(self):
        logits = Tensor(np.array([[2.0, 0.5, -1.0], [-1.0, 3.0, 0.0]], dtype=np.float32), requires_grad=True)
        targets = Tensor(np.array([0, 1], dtype=np.int64), dtype="int64")
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert float(loss.numpy()[0]) > 0.0
