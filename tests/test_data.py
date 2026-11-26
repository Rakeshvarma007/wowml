"""Tests for Dataset, TensorDataset, Sampler, and DataLoader."""
import numpy as np
from wowml import Tensor
from wowml.data import DataLoader, Dataset


class _ListDataset(Dataset):
    def __init__(self, n=10):
        self.x = np.arange(n, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return float(self.x[idx])


def test_dataloader_batches():
    ds = _ListDataset(10)
    dl = DataLoader(ds, batch_size=3, shuffle=False, drop_last=False)
    batches = list(dl)
    assert len(batches) == 4
    assert batches[0] == [0.0, 1.0, 2.0]
    assert batches[-1] == [9.0]

    dl_drop = DataLoader(ds, batch_size=3, shuffle=True, drop_last=True)
    batches_drop = list(dl_drop)
    assert len(batches_drop) == 3
