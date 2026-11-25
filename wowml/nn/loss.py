"""
loss.py — Loss function modules.
"""
from wowml.nn.module import Module
from wowml.tensor import Tensor


class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError

class L1Loss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

class BCELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

class BCEWithLogitsLoss(Module):
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError
