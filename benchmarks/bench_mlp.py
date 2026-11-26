"""
bench_mlp.py — End-to-end MLP training step benchmark (forward + backward + optimizer step).
"""
import statistics
import time


def _measure(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def benchmark_mlp(batch_size: int = 128, in_dim: int = 784, hidden_dim: int = 512, out_dim: int = 10):
    import numpy as np
    import wowml
    import wowml.nn as nn
    import wowml.optim as optim

    x_np = np.random.randn(batch_size, in_dim).astype(np.float32)
    y_np = np.random.randn(batch_size, out_dim).astype(np.float32)

    try:
        import torch
        pt_model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )
        pt_opt = torch.optim.Adam(pt_model.parameters(), lr=1e-3)
        pt_loss_fn = torch.nn.MSELoss()
        x_pt = torch.from_numpy(x_np)
        y_pt = torch.from_numpy(y_np)

        def step_pt():
            pt_opt.zero_grad()
            loss = pt_loss_fn(pt_model(x_pt), y_pt)
            loss.backward()
            pt_opt.step()

        print(f"PyTorch MLP train step: {_measure(step_pt):.3f} ms")
    except ImportError:
        pass

    try:
        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        x_w = wowml.Tensor.from_numpy(x_np)
        y_w = wowml.Tensor.from_numpy(y_np)

        def step_wowml():
            opt.zero_grad()
            loss = loss_fn(model(x_w), y_w)
            loss.backward()
            opt.step()

        print(f"WowML   MLP train step: {_measure(step_wowml):.3f} ms")
    except (NotImplementedError, RuntimeError) as e:
        print(f"WowML   MLP train step: skipped ({e})")


if __name__ == "__main__":
    benchmark_mlp()
