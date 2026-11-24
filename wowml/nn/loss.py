"""
loss.py — Loss function modules.
"""
from wowml.nn.module import Module
from wowml.tensor import Tensor


class MSELoss(Module):
    """Mean Squared Error loss."""
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class CrossEntropyLoss(Module):
    """Cross-entropy loss (log-softmax + NLL)."""
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError
