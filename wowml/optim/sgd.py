"""
sgd.py — Stochastic Gradient Descent optimizer.
"""
from wowml.optim.optimizer import Optimizer


class SGD(Optimizer):
    """SGD with optional momentum and weight decay."""

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay})
        self.state = {}

    def step(self):
        # TODO: implement SGD update rule
        raise NotImplementedError
