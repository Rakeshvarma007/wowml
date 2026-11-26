"""
graph.py — Computation graph utilities, parameter counts, and Graphviz DOT export.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wowml.tensor import Tensor


def trace_graph(tensor: "Tensor") -> str:
    """Return a human-readable text representation of the computational DAG."""
    lines = []
    visited = set()

    def _walk(node: "Tensor", depth: int) -> None:
        nid = id(node)
        op_name = node._grad_fn.__class__.__name__ if node._grad_fn else "Leaf"
        indent = "  " * depth
        lines.append(f"{indent}{op_name} shape={node.shape} req_grad={node.requires_grad}")
        if nid in visited or node._grad_fn is None:
            return
        visited.add(nid)
        for p in node._grad_fn.inputs:
            if hasattr(p, "_grad_fn"):
                _walk(p, depth + 1)

    _walk(tensor, 0)
    return "\n".join(lines)


def count_parameters(tensor: "Tensor") -> int:
    """Count total leaf parameters reachable in the computational graph."""
    visited = set()
    total = 0

    def _walk(node: "Tensor") -> None:
        nonlocal total
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)
        if node._grad_fn is None:
            if node.requires_grad:
                total += node.numel
        else:
            for p in node._grad_fn.inputs:
                if hasattr(p, "_grad_fn"):
                    _walk(p)

    _walk(tensor)
    return total


def export_dot(root: "Tensor", filename: Optional[str] = None) -> str:
    """Export the computational DAG to Graphviz DOT format."""
    dot_lines = ["digraph WowMLGraph {", "  rankdir=BT;"]
    visited = set()

    def _walk(node: "Tensor") -> None:
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)
        label = node._grad_fn.__class__.__name__ if node._grad_fn else f"Leaf\\n{node.shape}"
        shape = "box" if node._grad_fn else "ellipse"
        dot_lines.append(f'  n{nid} [label="{label}", shape={shape}];')
        if node._grad_fn is not None:
            for p in node._grad_fn.inputs:
                if hasattr(p, "_grad_fn"):
                    dot_lines.append(f"  n{id(p)} -> n{nid};")
                    _walk(p)

    _walk(root)
    dot_lines.append("}")
    dot_str = "\n".join(dot_lines)
    if filename is not None:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(dot_str)
    return dot_str
