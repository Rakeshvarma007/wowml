"""
engine.py — Reverse-mode automatic differentiation engine.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wowml.tensor import Tensor

_GRAD_ENABLED = True


@contextmanager
def no_grad():
    """Context manager disabling autograd graph tracking."""
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def is_grad_enabled() -> bool:
    """Return whether gradient tracking is currently active."""
    return _GRAD_ENABLED


def _topological_sort(root: "Tensor") -> List["Tensor"]:
    """Return computation graph nodes in reverse topological order (root first)."""
    visited = set()
    order: List["Tensor"] = []

    def _visit(node: "Tensor") -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        if node._grad_fn is not None:
            for parent in node._grad_fn.inputs:
                if hasattr(parent, "_grad_fn"):
                    _visit(parent)
        order.append(node)

    _visit(root)
    return list(reversed(order))


def backward(tensor: "Tensor", gradient: Optional["Tensor"] = None, retain_graph: bool = False) -> None:
    """
    Run reverse-mode automatic differentiation starting from tensor.
    Accumulates gradients into .grad on leaf nodes.
    """
    from wowml.tensor import Tensor

    if not tensor.requires_grad:
        return

    if gradient is None:
        if tensor.numel != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs.")
        gradient = Tensor.ones(*tensor.shape, dtype=tensor.dtype)

    grads: Dict[int, Tensor] = {id(tensor): gradient}
    nodes = _topological_sort(tensor)

    for node in nodes:
        node_grad = grads.get(id(node))
        if node_grad is None:
            continue

        if node._grad_fn is None:
            if node.requires_grad:
                if node.grad is None:
                    node.grad = node_grad
                else:
                    with no_grad():
                        node.grad = node.grad + node_grad
            continue

        fn = node._grad_fn
        with no_grad():
            input_grads = fn.backward(node_grad)
        if not isinstance(input_grads, tuple):
            input_grads = (input_grads,)

        for parent, p_grad in zip(fn.inputs, input_grads):
            if p_grad is None or not isinstance(parent, Tensor) or not parent.requires_grad:
                continue
            pid = id(parent)
            if pid not in grads:
                grads[pid] = p_grad
            else:
                with no_grad():
                    grads[pid] = grads[pid] + p_grad

        if not retain_graph:
            node._grad_fn = None


def grad(
    outputs: Sequence["Tensor"],
    inputs: Sequence["Tensor"],
    grad_outputs: Optional[Sequence["Tensor"]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Tuple["Tensor", ...]:
    """
    Compute and return the sum of gradients of outputs with respect to inputs.
    """
    from wowml.tensor import Tensor

    if isinstance(outputs, Tensor):
        outputs = (outputs,)
    if isinstance(inputs, Tensor):
        inputs = (inputs,)

    if grad_outputs is None:
        grad_outputs = tuple(Tensor.ones(*o.shape, dtype=o.dtype) for o in outputs)

    grads: Dict[int, Tensor] = {}
    for out, g in zip(outputs, grad_outputs):
        grads[id(out)] = g

    visited = set()
    order: List[Tensor] = []

    def _visit(node: Tensor) -> None:
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)
        if node._grad_fn is not None:
            for p in node._grad_fn.inputs:
                if isinstance(p, Tensor):
                    _visit(p)
        order.append(node)

    for out in outputs:
        _visit(out)

    for node in reversed(order):
        node_grad = grads.get(id(node))
        if node_grad is None or node._grad_fn is None:
            continue
        fn = node._grad_fn
        if create_graph:
            input_grads = fn.backward(node_grad)
        else:
            with no_grad():
                input_grads = fn.backward(node_grad)
        if not isinstance(input_grads, tuple):
            input_grads = (input_grads,)
        for parent, p_grad in zip(fn.inputs, input_grads):
            if p_grad is None or not isinstance(parent, Tensor) or not parent.requires_grad:
                continue
            pid = id(parent)
            if pid not in grads:
                grads[pid] = p_grad
            else:
                if create_graph:
                    grads[pid] = grads[pid] + p_grad
                else:
                    with no_grad():
                        grads[pid] = grads[pid] + p_grad
        if not retain_graph and not create_graph:
            node._grad_fn = None

    return tuple(grads.get(id(inp), Tensor.zeros(*inp.shape, dtype=inp.dtype)) for inp in inputs)
