"""
rmsprop.py — RMSprop optimizer.
"""
import numpy as np
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

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            mu = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                p_arr = p.numpy()
                g = p.grad.numpy()
                if wd != 0.0:
                    g = g + wd * p_arr

                st = self.state[id(p)]
                if "square_avg" not in st:
                    st["square_avg"] = np.zeros_like(p_arr)
                    if mu > 0.0:
                        st["buf"] = np.zeros_like(p_arr)

                sq = st["square_avg"]
                sq[:] = alpha * sq + (1.0 - alpha) * (g * g)
                avg = np.sqrt(sq) + eps

                if mu > 0.0:
                    buf = st["buf"]
                    buf[:] = mu * buf + g / avg
                    p_arr -= lr * buf
                else:
                    p_arr -= lr * g / avg
