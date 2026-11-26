"""
dataset.py — Abstract Dataset and concrete TensorDataset classes.
"""
from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from wowml.tensor import Tensor


class Dataset:
    def __len__(self) -> int: raise NotImplementedError
    def __getitem__(self, index: int): raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping tensors (e.g. features and targets)."""

    def __init__(self, *tensors: Tensor):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor.")
        first_len = tensors[0].shape[0]
        for t in tensors[1:]:
            if t.shape[0] != first_len:
                raise ValueError(f"Size mismatch between tensors: {first_len} vs {t.shape[0]}")
        self.tensors: Tuple[Tensor, ...] = tensors

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        return tuple(t[index] for t in self.tensors)
