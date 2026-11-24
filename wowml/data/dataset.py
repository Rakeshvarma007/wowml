"""
dataset.py — Dataset base class.
"""


class Dataset:
    """Abstract base class. Subclasses implement __len__ and __getitem__."""
    def __len__(self) -> int:        raise NotImplementedError
    def __getitem__(self, index):    raise NotImplementedError
