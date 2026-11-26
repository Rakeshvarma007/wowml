"""
embedding.py — Lookup table embedding layer.
"""
import numpy as np
from wowml.nn.module import Module
from wowml.nn.parameter import Parameter
from wowml.tensor import Tensor


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: str = "float32"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w_arr = np.random.randn(num_embeddings, embedding_dim).astype(dtype)
        self.weight = Parameter(w_arr, dtype=dtype)

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices.numpy().astype(np.int64)
        out = self.weight.numpy()[idx]
        return Tensor(np.ascontiguousarray(out), dtype=self.weight.dtype)

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
