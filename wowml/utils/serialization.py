"""
serialization.py — High-performance binary model and tensor serialization without pickle.
Format:
  - 8 bytes magic: b"WOWML01\\x00"
  - 8 bytes uint64: JSON header length N
  - N bytes: UTF-8 JSON metadata (keys -> {dtype, shape, offset, nbytes})
  - Raw contiguous tensor byte buffers
"""
from __future__ import annotations
import json
import pathlib
import struct
from collections import OrderedDict
from typing import Any, BinaryIO, Union

import numpy as np
from wowml.tensor import Tensor

WOWML_MAGIC = b"WOWML01\x00"


def _encode_tensor(t: Tensor, buffer_chunks: list, cur_offset: int):
    arr = np.ascontiguousarray(t.numpy())
    raw = arr.tobytes()
    meta = {
        "dtype": t.dtype,
        "shape": list(t.shape),
        "offset": cur_offset,
        "nbytes": len(raw),
    }
    buffer_chunks.append(raw)
    return meta, cur_offset + len(raw)


def save(obj: Any, path_or_file: Union[str, pathlib.Path, BinaryIO]) -> None:
    """
    Serialize a Tensor or state_dict (dict of Tensors) to a standalone binary file.
    Does not use pickle.
    """
    buffer_chunks = []
    cur_offset = 0

    if isinstance(obj, Tensor):
        meta, cur_offset = _encode_tensor(obj, buffer_chunks, cur_offset)
        header = {"kind": "tensor", "tensor": meta}
    elif isinstance(obj, dict):
        tensors_meta = OrderedDict()
        for k, v in obj.items():
            if not isinstance(v, Tensor):
                raise TypeError(f"Unsupported value type in state_dict for '{k}': {type(v)}")
            meta, cur_offset = _encode_tensor(v, buffer_chunks, cur_offset)
            tensors_meta[k] = meta
        header = {"kind": "state_dict", "tensors": tensors_meta}
    else:
        raise TypeError(f"Unsupported object type for wowml.save: {type(obj)}")

    header_bytes = json.dumps(header).encode("utf-8")

    def _write_to(fh: BinaryIO):
        fh.write(WOWML_MAGIC)
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for chunk in buffer_chunks:
            fh.write(chunk)

    if hasattr(path_or_file, "write"):
        _write_to(path_or_file)
    else:
        with open(path_or_file, "wb") as f:
            _write_to(f)


def load(path_or_file: Union[str, pathlib.Path, BinaryIO]) -> Any:
    """Deserialize a Tensor or state_dict from a WowML binary file."""
    def _read_from(fh: BinaryIO):
        magic = fh.read(len(WOWML_MAGIC))
        if magic != WOWML_MAGIC:
            raise ValueError("Invalid WowML binary file (bad magic header).")
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len).decode("utf-8"))
        payload = fh.read()

        def _decode_tensor(meta: dict) -> Tensor:
            start = meta["offset"]
            end = start + meta["nbytes"]
            arr = np.frombuffer(payload[start:end], dtype=meta["dtype"]).reshape(meta["shape"]).copy()
            return Tensor(arr, dtype=meta["dtype"])

        if header["kind"] == "tensor":
            return _decode_tensor(header["tensor"])
        if header["kind"] == "state_dict":
            out = OrderedDict()
            for k, meta in header["tensors"].items():
                out[k] = _decode_tensor(meta)
            return out
        raise ValueError(f"Unknown WowML archive kind: {header['kind']}")

    if hasattr(path_or_file, "read"):
        return _read_from(path_or_file)
    with open(path_or_file, "rb") as f:
        return _read_from(f)
