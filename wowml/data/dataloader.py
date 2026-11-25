"""
dataloader.py — DataLoader supporting multi-worker loading, shuffling, and drop_last.
"""
from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from wowml.data.dataset import Dataset
    from wowml.data.sampler import Sampler


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
        self.sampler = sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def __iter__(self):
        # TODO: Implement batch generator with optional multi-worker multiprocessing stub
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
