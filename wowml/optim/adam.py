"""
adam.py — Adam optimizer with bias correction.
"""
import numpy as np
from wowml.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

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
                    st["m"] = np.zeros_like(p_arr)
                    st["v"] = np.zeros_like(p_arr)

                st["step"] += 1
                t = st["step"]
                m, v = st["m"], st["v"]
                m[:] = beta1 * m + (1.0 - beta1) * g
                v[:] = beta2 * v + (1.0 - beta2) * (g * g)

                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)
                p_arr -= lr * m_hat / (np.sqrt(v_hat) + eps)
