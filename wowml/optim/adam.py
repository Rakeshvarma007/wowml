"""
adam.py — Adam optimizer.
"""
from wowml.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer with bias correction."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        self.state = {}

    def step(self):
        # TODO: implement Adam update rule
        raise NotImplementedError
