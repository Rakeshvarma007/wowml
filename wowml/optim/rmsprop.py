"""
rmsprop.py — RMSprop optimizer.
"""
from wowml.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0):
        defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        # TODO: Implement RMSprop update stub
        raise NotImplementedError
