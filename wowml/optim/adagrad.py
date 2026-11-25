"""
adagrad.py — Adagrad optimizer.
"""
from wowml.optim.optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0.0,
                 weight_decay: float = 0.0, initial_accumulator_value: float = 0.0, eps: float = 1e-10):
        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "eps": eps,
        }
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        # TODO: Implement Adagrad update stub
        raise NotImplementedError
