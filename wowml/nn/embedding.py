"""
embedding.py — Lookup table embedding layer.
"""
from wowml.nn.module import Module
from wowml.tensor import Tensor


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # TODO: Initialize embedding weight matrix stub

    def forward(self, indices: Tensor) -> Tensor:
        raise NotImplementedError
