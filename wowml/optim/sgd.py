"""
sgd.py — Stochastic Gradient Descent with momentum, weight decay, and Nesterov acceleration.
"""
from wowml.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        # TODO: Implement SGD step stub with momentum and Nesterov
        raise NotImplementedError
