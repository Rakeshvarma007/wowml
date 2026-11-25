"""
serialization.py — High-performance binary model and tensor serialization without pickle.
Implements custom binary format: magic header (b"WOWML01"), metadata header, and raw byte buffers.
"""
from __future__ import annotations
from typing import Any, BinaryIO, Dict, Union
import pathlib

WOWML_MAGIC = b"WOWML01\x00"


def save(obj: Any, path_or_file: Union[str, pathlib.Path, BinaryIO]) -> None:
    """
    Serialize a Tensor, state_dict, or model to a secure, standalone binary file.
    Does not use pickle.
    """
    # TODO: Implement binary serialization format stub
    raise NotImplementedError


def load(path_or_file: Union[str, pathlib.Path, BinaryIO]) -> Any:
    """
    Deserialize a Tensor, state_dict, or model from a WowML binary file.
    """
    # TODO: Implement binary deserialization stub
    raise NotImplementedError
