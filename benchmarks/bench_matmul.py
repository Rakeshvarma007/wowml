"""
bench_matmul.py — Matrix multiplication GFLOPS & latency benchmark vs PyTorch, JAX, and NumPy.
"""
import statistics
import time
from typing import Callable, Tuple


def _measure(fn: Callable, warmup: int = 5, iters: int = 20) -> Tuple[float, float]:
    """Return (median_ms, min_ms) for fn()."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return statistics.median(times), min(times)


def benchmark_matmul(sizes=(128, 512, 1024, 2048)):
    import numpy as np
    import wowml

    has_torch = False
    try:
        import torch
        has_torch = True
    except ImportError:
        pass

    has_jax = False
    try:
        import jax
        import jax.numpy as jnp
        has_jax = True
    except ImportError:
        pass

    print(f"{'Size':>10} | {'Backend':<10} | {'Median (ms)':>12} | {'GFLOPS':>10}")
    print("-" * 52)

    for n in sizes:
        flops = 2.0 * (n ** 3)
        a_np = np.random.randn(n, n).astype(np.float32)
        b_np = np.random.randn(n, n).astype(np.float32)

        # NumPy
        med, _ = _measure(lambda: a_np @ b_np)
        print(f"{f'{n}x{n}':>10} | {'NumPy':<10} | {med:>12.3f} | {(flops / (med * 1e6)):>10.2f}")

        # PyTorch
        if has_torch:
            a_pt = torch.from_numpy(a_np)
            b_pt = torch.from_numpy(b_np)
            med, _ = _measure(lambda: a_pt @ b_pt)
            print(f"{f'{n}x{n}':>10} | {'PyTorch':<10} | {med:>12.3f} | {(flops / (med * 1e6)):>10.2f}")

        # JAX
        if has_jax:
            a_jx = jnp.asarray(a_np)
            b_jx = jnp.asarray(b_np)
            med, _ = _measure(lambda: (a_jx @ b_jx).block_until_ready())
            print(f"{f'{n}x{n}':>10} | {'JAX':<10} | {med:>12.3f} | {(flops / (med * 1e6)):>10.2f}")

        # WowML
        try:
            a_w = wowml.Tensor.from_numpy(a_np)
            b_w = wowml.Tensor.from_numpy(b_np)
            med, _ = _measure(lambda: a_w @ b_w)
            print(f"{f'{n}x{n}':>10} | {'WowML':<10} | {med:>12.3f} | {(flops / (med * 1e6)):>10.2f}")
        except NotImplementedError:
            print(f"{f'{n}x{n}':>10} | {'WowML':<10} | {'NOT IMPL':>12} | {'-':>10}")
        except RuntimeError as e:
            print(f"{f'{n}x{n}':>10} | {'WowML':<10} | {'NO LIB':>12} | {'-':>10}")

        print("-" * 52)


if __name__ == "__main__":
    benchmark_matmul()
