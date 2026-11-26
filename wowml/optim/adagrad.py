"""
adagrad.py — Adagrad optimizer.
"""
import numpy as np
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

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            wd = group["weight_decay"]
            init_acc = group["initial_accumulator_value"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                p_arr = p.numpy()
                g = p.grad.numpy()
                if wd != 0.0:
                    g = g + wd * p_arr

                st = self.state[id(p)]
                if "step" not in st:
                    st["step"] = 0
                    st["sum"] = np.full_like(p_arr, init_acc)

                st["step"] += 1
                clr = lr / (1.0 + (st["step"] - 1) * lr_decay)
                st["sum"] += g * g
                p_arr -= clr * g / (np.sqrt(st["sum"]) + eps)
