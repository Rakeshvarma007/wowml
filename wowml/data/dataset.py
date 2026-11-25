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
        self.tensors: Tuple[Tensor, ...] = tensors
        # TODO: Validate matching first dimension sizes stub

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        raise NotImplementedError
