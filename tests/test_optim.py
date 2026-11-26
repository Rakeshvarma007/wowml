"""Tests for optimizers and learning rate schedulers."""
import numpy as np
import pytest
import wowml.nn as nn
import wowml.optim as optim
from wowml import Tensor


class TestOptimizers:
    def test_sgd_converges(self):
        w = nn.Parameter(np.array([5.0], dtype=np.float32))
        opt = optim.SGD([w], lr=0.1, momentum=0.9)
        for _ in range(150):
            opt.zero_grad()
            loss = (w * w).sum()
            loss.backward()
            opt.step()
        assert abs(float(w.numpy()[0])) < 1e-2

    def test_adam_converges(self):
        w = nn.Parameter(np.array([5.0, -3.0], dtype=np.float32))
        opt = optim.Adam([w], lr=0.2)
        for _ in range(200):
            opt.zero_grad()
            loss = (w * w).sum()
            loss.backward()
            opt.step()
        np.testing.assert_allclose(w.numpy(), [0.0, 0.0], atol=1e-2)

    def test_lr_scheduler(self):
        w = nn.Parameter(np.array([1.0], dtype=np.float32))
        opt = optim.SGD([w], lr=0.1)
        sched = optim.StepLR(opt, step_size=2, gamma=0.5)
        sched.step()  # epoch 0 -> 0.1
        assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-6
        sched.step()  # epoch 1 -> 0.1
        sched.step()  # epoch 2 -> 0.05
        assert abs(opt.param_groups[0]["lr"] - 0.05) < 1e-6
