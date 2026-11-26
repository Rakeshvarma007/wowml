"""
lr_scheduler.py — Learning rate schedulers.
"""
import math
from wowml.optim.optimizer import Optimizer


class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        self.last_epoch += 1
        for group, new_lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = new_lr

    def get_lr(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return list(self.base_lrs)
        if self.last_epoch % self.step_size == 0:
            return [g["lr"] * self.gamma for g in self.optimizer.param_groups]
        return [g["lr"] for g in self.optimizer.param_groups]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * t / self.T_max))
            for base_lr in self.base_lrs
        ]
