"""Tests for custom binary serialization without pickle."""
import numpy as np
import wowml
from wowml import Tensor
import wowml.nn as nn


def test_save_load_tensor(tmp_path):
    path = tmp_path / "tensor.wml"
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    wowml.utils.save(t, path)
    loaded = wowml.utils.load(path)
    np.testing.assert_allclose(loaded.numpy(), t.numpy())


def test_save_load_state_dict(tmp_path):
    path = tmp_path / "model.wml"
    m1 = nn.Linear(3, 2)
    wowml.utils.save(m1.state_dict(), path)

    m2 = nn.Linear(3, 2)
    m2.load_state_dict(wowml.utils.load(path))
    np.testing.assert_allclose(m1.weight.numpy(), m2.weight.numpy())
    np.testing.assert_allclose(m1.bias.numpy(), m2.bias.numpy())
