"""
adamw.py — AdamW optimizer (decoupled weight decay).
"""
from wowml.optim.optimizer import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        # TODO: Implement AdamW decoupled weight decay update stub
        raise NotImplementedError
