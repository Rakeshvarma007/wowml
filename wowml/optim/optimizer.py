"""
optimizer.py — Base optimizer class with param_groups, zero_grad, and state_dict support.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, Iterable


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params: Iterable, defaults: Dict[str, Any]):
        self.defaults = defaults
        self.state: Dict[Any, Dict[str, Any]] = defaultdict(dict)
        self.param_groups = [{"params": list(params), **defaults}]

    def zero_grad(self) -> None:
        """Zero out gradients for all managed parameters."""
        for group in self.param_groups:
            for p in group["params"]:
                p.zero_grad()

    def step(self) -> None:
        """Perform a single optimization step. Must be implemented by subclasses."""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {
            "state": dict(self.state),
            "param_groups": [
                {k: v for k, v in g.items() if k != "params"}
                for g in self.param_groups
            ],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.state.clear()
        self.state.update(state_dict.get("state", {}))
        for group, saved_group in zip(self.param_groups, state_dict.get("param_groups", [])):
            for k, v in saved_group.items():
                group[k] = v
