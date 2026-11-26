"""
sampler.py — Sampling strategies for dataset iteration.
"""
from __future__ import annotations
import random
from typing import Iterator, Sized


class Sampler:
    def __iter__(self) -> Iterator[int]: raise NotImplementedError
    def __len__(self) -> int: raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source: Sized, replacement: bool = False, num_samples: int | None = None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            k = self._num_samples or n
            return iter(random.choices(range(n), k=k))
        indices = list(range(n))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self._num_samples or len(self.data_source)
