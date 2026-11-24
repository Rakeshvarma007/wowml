"""
dataloader.py — DataLoader for batching and iterating over datasets.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import random
if TYPE_CHECKING:
    from wowml.data.dataset import Dataset


class DataLoader:
    def __init__(self, dataset: "Dataset", batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    def _collate(self, batch):
        """TODO: stack samples into Tensors."""
        return batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
