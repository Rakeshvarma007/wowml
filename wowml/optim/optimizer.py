"""
optimizer.py — Base optimizer class.
"""
from __future__ import annotations
from typing import Iterable, Dict, Any


class Optimizer:
    def __init__(self, params: Iterable, defaults: Dict[str, Any]):
        self.param_groups = [{"params": list(params), **defaults}]

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.zero_grad()

    def step(self):
        raise NotImplementedError
