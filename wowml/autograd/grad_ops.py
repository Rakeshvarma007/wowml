"""
grad_ops.py — All 17 differentiable operations wired to the C kernel.
"""
from __future__ import annotations
import ctypes
from typing import Tuple

from wowml._C._bindings import CWowmlTensor, DTYPE_FROM_STR, check_status, make_c_shape, require_lib
from wowml.autograd.function import Function
from wowml.tensor import Tensor


def _alloc_like(shape: Tuple[int, ...], dtype: str) -> Tensor:
    c_lib = require_lib()
    t = CWowmlTensor()
    ndim, c_shape = make_c_shape(shape)
    check_status(c_lib.wowml_tensor_alloc(ctypes.byref(t), int(DTYPE_FROM_STR[dtype]), ndim, c_shape))
    return Tensor(dtype=dtype, _c_tensor=t)


def _broadcast_shape(sa: Tuple[int, ...], sb: Tuple[int, ...]) -> Tuple[int, ...]:
    ndim = max(len(sa), len(sb))
    out = []
    for i in range(ndim):
        da = sa[len(sa) - 1 - i] if i < len(sa) else 1
        db = sb[len(sb) - 1 - i] if i < len(sb) else 1
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            raise ValueError(f"Shapes {sa} and {sb} are not broadcastable.")
    return tuple(reversed(out))


def _unbroadcast(grad: Tensor, target_shape: Tuple[int, ...]) -> Tensor:
    """Sum out dimensions added or expanded by broadcasting."""
    if grad.shape == target_shape:
        return grad
    import numpy as np
    g_np = grad.numpy()
    while g_np.ndim > len(target_shape):
        g_np = g_np.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1 and g_np.shape[i] != 1:
            g_np = g_np.sum(axis=i, keepdims=True)
    return Tensor(np.ascontiguousarray(g_np), dtype=grad.dtype)


class Add(Function):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.save_for_backward(a.shape, b.shape)
        out_shape = _broadcast_shape(a.shape, b.shape)
        out = _alloc_like(out_shape, a.dtype)
        check_status(require_lib().wowml_add(out.c_ptr, a.c_ptr, b.c_ptr))
        return out

    def backward(self, grad_output: Tensor):
        sa, sb = self.saved_tensors
        return _unbroadcast(grad_output, sa), _unbroadcast(grad_output, sb)


class Sub(Function):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.save_for_backward(a.shape, b.shape)
        out_shape = _broadcast_shape(a.shape, b.shape)
        out = _alloc_like(out_shape, a.dtype)
        check_status(require_lib().wowml_sub(out.c_ptr, a.c_ptr, b.c_ptr))
        return out

    def backward(self, grad_output: Tensor):
        sa, sb = self.saved_tensors
        return _unbroadcast(grad_output, sa), _unbroadcast(-grad_output, sb)


class Mul(Function):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.save_for_backward(a, b)
        out_shape = _broadcast_shape(a.shape, b.shape)
        out = _alloc_like(out_shape, a.dtype)
        check_status(require_lib().wowml_mul(out.c_ptr, a.c_ptr, b.c_ptr))
        return out

    def backward(self, grad_output: Tensor):
        a, b = self.saved_tensors
        return _unbroadcast(grad_output * b, a.shape), _unbroadcast(grad_output * a, b.shape)


class Div(Function):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.save_for_backward(a, b)
        out_shape = _broadcast_shape(a.shape, b.shape)
        out = _alloc_like(out_shape, a.dtype)
        check_status(require_lib().wowml_div(out.c_ptr, a.c_ptr, b.c_ptr))
        return out

    def backward(self, grad_output: Tensor):
        a, b = self.saved_tensors
        ga = grad_output / b
        gb = -grad_output * a / (b ** 2.0)
        return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)


class Neg(Function):
    def forward(self, x: Tensor) -> Tensor:
        out = _alloc_like(x.shape, x.dtype)
        check_status(require_lib().wowml_neg(out.c_ptr, x.c_ptr))
        return out

    def backward(self, grad_output: Tensor):
        return (-grad_output,)


class MatMul(Function):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.save_for_backward(a, b)
        if a.ndim == 2 and b.ndim == 2:
            out = _alloc_like((a.shape[0], b.shape[1]), a.dtype)
            check_status(require_lib().wowml_matmul(out.c_ptr, a.c_ptr, b.c_ptr))
            return out
        elif a.ndim == 3 and b.ndim == 3:
            out = _alloc_like((a.shape[0], a.shape[1], b.shape[2]), a.dtype)
            check_status(require_lib().wowml_bmm(out.c_ptr, a.c_ptr, b.c_ptr))
            return out
        raise ValueError(f"MatMul expects 2D or 3D tensors, got {a.shape} and {b.shape}.")

    def backward(self, grad_output: Tensor):
        a, b = self.saved_tensors
        if a.ndim == 2:
            return grad_output @ b.T, a.T @ grad_output
        return grad_output @ b.transpose(1, 2), a.transpose(1, 2) @ grad_output


class Pow(Function):
    def forward(self, x: Tensor, exponent: float) -> Tensor:
        self.save_for_backward(x, exponent)
        out = _alloc_like(x.shape, x.dtype)
        check_status(require_lib().wowml_pow_op(out.c_ptr, x.c_ptr, float(exponent)))
        return out

    def backward(self, grad_output: Tensor):
        x, exp_val = self.saved_tensors
        return grad_output * (exp_val * (x ** (exp_val - 1.0))), None


class Sqrt(Function):
    def forward(self, x: Tensor) -> Tensor:
        out = _alloc_like(x.shape, x.dtype)
        check_status(require_lib().wowml_sqrt(out.c_ptr, x.c_ptr))
        self.save_for_backward(out)
        return out

    def backward(self, grad_output: Tensor):
        (out,) = self.saved_tensors
        return (grad_output / (out * 2.0),)


class Exp(Function):
    def forward(self, x: Tensor) -> Tensor:
        out = _alloc_like(x.shape, x.dtype)
        check_status(require_lib().wowml_exp(out.c_ptr, x.c_ptr))
        self.save_for_backward(out)
        return out

    def backward(self, grad_output: Tensor):
        (out,) = self.saved_tensors
        return (grad_output * out,)


class Log(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.save_for_backward(x)
        out = _alloc_like(x.shape, x.dtype)
        check_status(require_lib().wowml_log(out.c_ptr, x.c_ptr))
        return out

    def backward(self, grad_output: Tensor):
        (x,) = self.saved_tensors
        return (grad_output / x,)


class ReLU(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.save_for_backward(x)
        out = _alloc_like(x.shape, x.dtype)
        check_status(require_lib().wowml_relu(out.c_ptr, x.c_ptr))
        return out

    def backward(self, grad_output: Tensor):
        (x,) = self.saved_tensors
        import numpy as np
        mask = Tensor((x.numpy() > 0).astype(x.dtype))
        return (grad_output * mask,)


class Sigmoid(Function):
    def forward(self, x: Tensor) -> Tensor:
        out = _alloc_like(x.shape, x.dtype)
        check_status(require_lib().wowml_sigmoid(out.c_ptr, x.c_ptr))
        self.save_for_backward(out)
        return out

    def backward(self, grad_output: Tensor):
        (s,) = self.saved_tensors
        return (grad_output * s * (1.0 - s),)


class Tanh(Function):
    def forward(self, x: Tensor) -> Tensor:
        out = _alloc_like(x.shape, x.dtype)
        check_status(require_lib().wowml_tanh_op(out.c_ptr, x.c_ptr))
        self.save_for_backward(out)
        return out

    def backward(self, grad_output: Tensor):
        (t,) = self.saved_tensors
        return (grad_output * (1.0 - t * t),)


class Sum(Function):
    def forward(self, x: Tensor, axis=None, keepdims: bool = False) -> Tensor:
        self.save_for_backward(x.shape, axis, keepdims)
        if axis is None:
            out = _alloc_like((1,), x.dtype)
            check_status(require_lib().wowml_sum(out.c_ptr, x.c_ptr, -1, 0))
            return out
        ax = axis if axis >= 0 else axis + x.ndim
        out_shape = tuple(
            1 if (i == ax and keepdims) else s
            for i, s in enumerate(x.shape)
            if (i != ax or keepdims)
        ) or (1,)
        out = _alloc_like(out_shape, x.dtype)
        check_status(require_lib().wowml_sum(out.c_ptr, x.c_ptr, ax, int(keepdims)))
        return out

    def backward(self, grad_output: Tensor):
        orig_shape, axis, keepdims = self.saved_tensors
        import numpy as np
        g = grad_output.numpy()
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis=axis)
        return Tensor(np.broadcast_to(g, orig_shape).copy(), dtype=grad_output.dtype), None, None


class Mean(Function):
    def forward(self, x: Tensor, axis=None, keepdims: bool = False) -> Tensor:
        self.save_for_backward(x.shape, axis, keepdims)
        if axis is None:
            out = _alloc_like((1,), x.dtype)
            check_status(require_lib().wowml_mean(out.c_ptr, x.c_ptr, -1, 0))
            return out
        ax = axis if axis >= 0 else axis + x.ndim
        out_shape = tuple(
            1 if (i == ax and keepdims) else s
            for i, s in enumerate(x.shape)
            if (i != ax or keepdims)
        ) or (1,)
        out = _alloc_like(out_shape, x.dtype)
        check_status(require_lib().wowml_mean(out.c_ptr, x.c_ptr, ax, int(keepdims)))
        return out

    def backward(self, grad_output: Tensor):
        orig_shape, axis, keepdims = self.saved_tensors
        import numpy as np
        g = grad_output.numpy()
        if axis is None:
            scale = 1.0 / float(np.prod(orig_shape))
        else:
            ax = axis if axis >= 0 else axis + len(orig_shape)
            scale = 1.0 / float(orig_shape[ax])
            if not keepdims:
                g = np.expand_dims(g, axis=ax)
        return Tensor((np.broadcast_to(g, orig_shape) * scale).astype(grad_output.dtype)), None, None


class Reshape(Function):
    def forward(self, x: Tensor, shape: Tuple[int, ...]) -> Tensor:
        self.save_for_backward(x.shape)
        c_lib = require_lib()
        xc = x.contiguous()
        t = CWowmlTensor()
        check_status(c_lib.wowml_tensor_clone(ctypes.byref(t), xc.c_ptr))
        ndim, c_shape = make_c_shape(shape)
        check_status(c_lib.wowml_tensor_reshape(ctypes.byref(t), ndim, c_shape))
        return Tensor(dtype=x.dtype, _c_tensor=t)

    def backward(self, grad_output: Tensor):
        (orig_shape,) = self.saved_tensors
        return grad_output.reshape(*orig_shape), None


class Transpose(Function):
    def forward(self, x: Tensor, dim0: int, dim1: int) -> Tensor:
        self.save_for_backward(dim0, dim1)
        c_lib = require_lib()
        view = CWowmlTensor()
        check_status(c_lib.wowml_tensor_transpose(ctypes.byref(view), x.c_ptr, dim0, dim1))
        out = Tensor(dtype=x.dtype, _c_tensor=view)
        out._np_ref = x
        return out

    def backward(self, grad_output: Tensor):
        dim0, dim1 = self.saved_tensors
        return grad_output.transpose(dim0, dim1), None, None
