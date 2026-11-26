"""
conv.py — 2D Convolution layer using C-level im2col + matmul.
"""
import math
import ctypes
from wowml._C._bindings import CWowmlTensor, DTYPE_FROM_STR, check_status, make_c_shape, require_lib
from wowml.nn.module import Module
from wowml.nn.parameter import Parameter
from wowml.tensor import Tensor


class Conv2d(Module):
    """2D convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: str = "float32",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        k = kernel_size
        bound = 1.0 / math.sqrt(in_channels * k * k)
        import numpy as np
        w_arr = np.random.uniform(-bound, bound, (out_channels, in_channels, k, k)).astype(dtype)
        self.weight = Parameter(w_arr, dtype=dtype)
        if bias:
            b_arr = np.random.uniform(-bound, bound, (out_channels,)).astype(dtype)
            self.bias = Parameter(b_arr, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        k = self.kernel_size
        out_h = (H + 2 * self.padding - k) // self.stride + 1
        out_w = (W + 2 * self.padding - k) // self.stride + 1

        c_lib = require_lib()
        col_c = CWowmlTensor()
        ndim, c_shape = make_c_shape((C * k * k, N * out_h * out_w))
        check_status(c_lib.wowml_tensor_alloc(ctypes.byref(col_c), int(DTYPE_FROM_STR[x.dtype]), ndim, c_shape))
        col = Tensor(dtype=x.dtype, _c_tensor=col_c)
        check_status(c_lib.wowml_im2col(col.c_ptr, x.c_ptr, k, k, self.padding, self.padding, self.stride, self.stride, 1, 1))

        w_flat = self.weight.reshape(self.out_channels, C * k * k)
        out_2d = w_flat @ col  # (out_channels, N * out_h * out_w)
        out = out_2d.reshape(self.out_channels, N, out_h, out_w).transpose(0, 1)
        if self.bias is not None:
            out = out + self.bias.reshape(1, self.out_channels, 1, 1)
        return out

    def __repr__(self) -> str:
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )
