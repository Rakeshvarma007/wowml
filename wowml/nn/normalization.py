"""
normalization.py — Batch normalization layers.
"""
from __future__ import annotations
import numpy as np
from wowml.nn.module import Module
from wowml.nn.parameter import Parameter
from wowml.tensor import Tensor


class _BatchNorm(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, dtype: str = "float32"):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = Parameter(np.ones(num_features, dtype=dtype), dtype=dtype)
            self.bias = Parameter(np.zeros(num_features, dtype=dtype), dtype=dtype)
        else:
            self.weight = None
            self.bias = None

        self.register_buffer("running_mean", Tensor(np.zeros(num_features, dtype=dtype), dtype=dtype))
        self.register_buffer("running_var", Tensor(np.ones(num_features, dtype=dtype), dtype=dtype))


class BatchNorm1d(_BatchNorm):
    """Batch Normalization over 2D (N, C) inputs."""
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            diff = x - mean
            var = (diff * diff).mean(axis=0, keepdims=True)
            rm = self.running_mean.numpy()
            rv = self.running_var.numpy()
            rm[:] = (1.0 - self.momentum) * rm + self.momentum * mean.numpy().reshape(-1)
            rv[:] = (1.0 - self.momentum) * rv + self.momentum * var.numpy().reshape(-1)
        else:
            mean = self.running_mean.reshape(1, self.num_features)
            var = self.running_var.reshape(1, self.num_features)
            diff = x - mean
        x_hat = diff / ((var + self.eps) ** 0.5)
        if self.affine:
            x_hat = x_hat * self.weight.reshape(1, self.num_features) + self.bias.reshape(1, self.num_features)
        return x_hat


class BatchNorm2d(_BatchNorm):
    """Batch Normalization over 4D (N, C, H, W) inputs."""
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x_np = x.numpy()
            mean_np = x_np.mean(axis=(0, 2, 3), keepdims=True)
            var_np = x_np.var(axis=(0, 2, 3), keepdims=True)
            rm = self.running_mean.numpy()
            rv = self.running_var.numpy()
            rm[:] = (1.0 - self.momentum) * rm + self.momentum * mean_np.reshape(-1)
            rv[:] = (1.0 - self.momentum) * rv + self.momentum * var_np.reshape(-1)
            mean = Tensor(mean_np.astype(x.dtype), dtype=x.dtype)
            var = Tensor(var_np.astype(x.dtype), dtype=x.dtype)
        else:
            mean = self.running_mean.reshape(1, self.num_features, 1, 1)
            var = self.running_var.reshape(1, self.num_features, 1, 1)
        x_hat = (x - mean) / ((var + self.eps) ** 0.5)
        if self.affine:
            x_hat = x_hat * self.weight.reshape(1, self.num_features, 1, 1) + self.bias.reshape(1, self.num_features, 1, 1)
        return x_hat
