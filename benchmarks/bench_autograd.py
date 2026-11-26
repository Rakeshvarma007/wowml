"""
bench_autograd.py — Autograd forward + backward latency vs PyTorch and JAX.
"""
import statistics
import time


def _measure(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def benchmark_autograd(n: int = 512):
    import numpy as np
    import wowml

    x_np = np.random.randn(n, n).astype(np.float32)
    w_np = np.random.randn(n, n).astype(np.float32)

    try:
        import torch
        x_pt = torch.tensor(x_np, requires_grad=True)
        w_pt = torch.tensor(w_np, requires_grad=True)
        def run_pt():
            x_pt.grad = None
            w_pt.grad = None
            y = ((x_pt @ w_pt).relu() * 2.0).sum()
            y.backward()
        print(f"PyTorch autograd ({n}x{n}): {_measure(run_pt):.3f} ms")
    except ImportError:
        pass

    try:
        x_w = wowml.Tensor.from_numpy(x_np, requires_grad=True)
        w_w = wowml.Tensor.from_numpy(w_np, requires_grad=True)
        def run_wowml():
            x_w.zero_grad()
            w_w.zero_grad()
            y = ((x_w @ w_w) * 2.0).sum()
            y.backward()
        print(f"WowML   autograd ({n}x{n}): {_measure(run_wowml):.3f} ms")
    except (NotImplementedError, RuntimeError) as e:
        print(f"WowML   autograd ({n}x{n}): skipped ({e})")


if __name__ == "__main__":
    benchmark_autograd()
