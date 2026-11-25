"""
normalization.py — Batch normalization layers.
"""
from __future__ import annotations
from wowml.nn.module import Module
from wowml.tensor import Tensor


class BatchNorm1d(Module):
    """Batch Normalization over 2D or 3D inputs."""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        # TODO: Initialize running_mean, running_var, weight, bias stubs

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class BatchNorm2d(Module):
    """Batch Normalization over 4D inputs (N, C, H, W)."""
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        # TODO: Initialize running_mean, running_var, weight, bias stubs

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
