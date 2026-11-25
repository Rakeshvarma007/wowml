"""
lr_scheduler.py — Learning rate schedulers.
"""
from wowml.optim.optimizer import Optimizer


class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def step(self):
        raise NotImplementedError

    def get_lr(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # TODO: Implement StepLR stub
        raise NotImplementedError


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # TODO: Implement CosineAnnealingLR stub
        raise NotImplementedError
