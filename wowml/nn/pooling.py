"""
pooling.py — 2D pooling layers (MaxPool2d and AvgPool2d).
"""
from __future__ import annotations
import numpy as np
from wowml.nn.module import Module
from wowml.tensor import Tensor


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        arr = x.numpy()
        if p > 0:
            arr = np.pad(arr, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant", constant_values=-np.inf)
        oh = (H + 2 * p - k) // s + 1
        ow = (W + 2 * p - k) // s + 1
        out = np.empty((N, C, oh, ow), dtype=arr.dtype)
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = arr[:, :, i * s:i * s + k, j * s:j * s + k].max(axis=(2, 3))
        return Tensor(out, dtype=x.dtype)


class AvgPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        arr = x.numpy()
        if p > 0:
            arr = np.pad(arr, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant", constant_values=0.0)
        oh = (H + 2 * p - k) // s + 1
        ow = (W + 2 * p - k) // s + 1
        out = np.empty((N, C, oh, ow), dtype=arr.dtype)
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = arr[:, :, i * s:i * s + k, j * s:j * s + k].mean(axis=(2, 3))
        return Tensor(out, dtype=x.dtype)
