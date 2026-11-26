"""
loss.py — Loss function modules.
"""
from wowml.autograd import grad_ops
from wowml.nn.module import Module
from wowml.tensor import Tensor


class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction - target
        return (diff * diff).mean()


class L1Loss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        diff = prediction - target
        return ((diff * diff) ** 0.5).mean()


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        import numpy as np
        # Numerically stable log-softmax + NLL
        B, C = logits.shape
        l_max = Tensor(logits.numpy().max(axis=1, keepdims=True).astype(logits.dtype), dtype=logits.dtype)
        shifted = logits - l_max
        log_sum_exp = grad_ops.Log.apply(grad_ops.Exp.apply(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp

        tgt_idx = targets.numpy().astype(np.int64)
        one_hot = np.zeros((B, C), dtype=logits.dtype)
        one_hot[np.arange(B), tgt_idx] = 1.0
        nll = -(log_probs * Tensor(one_hot, dtype=logits.dtype)).sum() / float(B)
        return nll


class BCELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        eps = 1e-7
        p = prediction + eps
        one_minus_p = (1.0 - prediction) + eps
        loss = -(target * grad_ops.Log.apply(p) + (1.0 - target) * grad_ops.Log.apply(one_minus_p))
        return loss.mean()


class BCEWithLogitsLoss(Module):
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = grad_ops.Sigmoid.apply(logits)
        return BCELoss().forward(probs, targets)
