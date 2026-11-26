"""
dataloader.py — DataLoader supporting batching, samplers, drop_last, and custom collate_fn.
"""
from __future__ import annotations
from typing import Callable, Optional, TYPE_CHECKING

from wowml.data.sampler import RandomSampler, SequentialSampler

if TYPE_CHECKING:
    from wowml.data.dataset import Dataset
    from wowml.data.sampler import Sampler


def _default_collate(batch):
    """Default collation: returns batch list (or tuples of batched items)."""
    if isinstance(batch[0], tuple):
        return tuple(list(col) for col in zip(*batch))
    return batch


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn or _default_collate
        self.drop_last = drop_last

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
