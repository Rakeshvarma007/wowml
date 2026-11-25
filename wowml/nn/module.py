"""
module.py — Base class for all neural network modules with state_dict serialization.
"""
from __future__ import annotations
from typing import Iterator, Tuple, Dict, Any, TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from wowml.nn.parameter import Parameter
    from wowml.tensor import Tensor


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self.training: bool = True

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any):
        from wowml.nn.parameter import Parameter
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_parameters", OrderedDict())[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", OrderedDict())[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> Iterator[Parameter]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            yield name, param
        for mod_name, module in self._modules.items():
            for pname, param in module.named_parameters():
                yield f"{mod_name}.{pname}", param

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """Return a dictionary containing references to whole state of module."""
        # TODO: Implement state_dict serialization stub
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Copy parameters and buffers from state_dict into this module."""
        # TODO: Implement load_state_dict stub
        raise NotImplementedError
