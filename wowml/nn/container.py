"""
container.py — Module containers.
"""
from __future__ import annotations
from typing import List
from wowml.nn.module import Module
from wowml.tensor import Tensor


class Sequential(Module):
    """Modules called in order."""
    def __init__(self, *modules: Module):
        super().__init__()
        for idx, mod in enumerate(modules):
            self._modules[str(idx)] = mod

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


class ModuleList(Module):
    """List of modules registered as sub-modules."""
    def __init__(self, modules: List[Module] | None = None):
        super().__init__()
        if modules:
            for idx, mod in enumerate(modules):
                self._modules[str(idx)] = mod

    def __len__(self):              return len(self._modules)
    def __getitem__(self, idx):     return list(self._modules.values())[idx]
    def append(self, module: Module):
        self._modules[str(len(self._modules))] = module
