"""
serialization.py — Save and load models / tensors.
"""
import pickle
from typing import Any


def save(obj: Any, path: str):
    """Serialize and save an object to disk."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path: str) -> Any:
    """Load a previously saved object."""
    with open(path, "rb") as f:
        return pickle.load(f)
