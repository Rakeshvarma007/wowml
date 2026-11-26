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

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                p_arr = p.numpy()
                g = p.grad.numpy()
                if wd != 0.0:
                    g = g + wd * p_arr
                if mu != 0.0:
                    st = self.state[id(p)]
                    if "velocity" not in st:
                        st["velocity"] = g.copy()
                    else:
                        st["velocity"][:] = mu * st["velocity"] + g
                    buf = st["velocity"]
                    update = (g + mu * buf) if nesterov else buf
                else:
                    update = g
                p_arr -= lr * update
