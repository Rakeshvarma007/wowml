"""
module.py — Base class for all neural network modules with state_dict serialization.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Any, Dict, Iterator, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wowml.nn.parameter import Parameter


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Any] = OrderedDict()
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

    def register_buffer(self, name: str, tensor: Any) -> None:
        """Register a non-parameter persistent buffer (e.g. BatchNorm running stats)."""
        self._buffers[name] = tensor
        super().__setattr__(name, tensor)

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

    def state_dict(self) -> OrderedDict[str, Any]:
        """Return an OrderedDict containing all parameters and buffers."""
        state = OrderedDict()
        for name, param in self._parameters.items():
            state[name] = param.detach()
        for name, buf in self._buffers.items():
            if buf is not None:
                state[name] = buf.detach() if hasattr(buf, "detach") else buf
        for mod_name, module in self._modules.items():
            for k, v in module.state_dict().items():
                state[f"{mod_name}.{k}"] = v
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Copy parameters and buffers from state_dict into this module."""
        own_state = OrderedDict()
        for name, param in self.named_parameters():
            own_state[name] = param

        for name, val in state_dict.items():
            if name in own_state:
                target = own_state[name]
                if hasattr(val, "numpy") and hasattr(target, "_c_tensor"):
                    # Copy data into target parameter
                    arr = val.numpy()
                    target.__init__(arr.copy(), dtype=target.dtype)
            elif strict:
                raise KeyError(f"Unexpected key '{name}' in state_dict.")

        if strict:
            missing = [k for k in own_state if k not in state_dict]
            if missing:
                raise KeyError(f"Missing key(s) in state_dict: {missing}")

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for name, mod in self._modules.items():
            mod_str = repr(mod).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines)
