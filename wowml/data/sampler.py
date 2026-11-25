"""
sampler.py — Sampling strategies for dataset iteration.
"""
from __future__ import annotations
from typing import Iterator, Sequence


class Sampler:
    def __iter__(self) -> Iterator[int]: raise NotImplementedError
    def __len__(self) -> int: raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source: Sequence):
        self.data_source = data_source
    def __iter__(self) -> Iterator[int]: raise NotImplementedError
    def __len__(self) -> int: raise NotImplementedError


class RandomSampler(Sampler):
    def __init__(self, data_source: Sequence, replacement: bool = False):
        self.data_source = data_source
        self.replacement = replacement
    def __iter__(self) -> Iterator[int]: raise NotImplementedError
    def __len__(self) -> int: raise NotImplementedError
