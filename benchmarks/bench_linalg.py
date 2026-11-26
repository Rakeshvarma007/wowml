"""
bench_linalg.py — Benchmark matrix decompositions (LU, QR, SVD, Cholesky, Solve) vs NumPy.
"""
import statistics
import time


def _measure(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def benchmark_linalg(n: int = 512):
    import numpy as np
    import wowml

    a_np = np.random.randn(n, n).astype(np.float64)
    spd_np = a_np @ a_np.T + np.eye(n) * 1e-2

    print(f"--- Linalg Benchmark ({n}x{n} float64) ---")
    print(f"NumPy cholesky: {_measure(lambda: np.linalg.cholesky(spd_np)):.3f} ms")
    print(f"NumPy qr:       {_measure(lambda: np.linalg.qr(a_np)):.3f} ms")
    print(f"NumPy svd:      {_measure(lambda: np.linalg.svd(a_np)):.3f} ms")

    try:
        spd_w = wowml.Tensor.from_numpy(spd_np)
        a_w = wowml.Tensor.from_numpy(a_np)
        print(f"WowML cholesky: {_measure(lambda: wowml.linalg.cholesky(spd_w)):.3f} ms")
        print(f"WowML qr:       {_measure(lambda: wowml.linalg.qr(a_w)):.3f} ms")
        print(f"WowML svd:      {_measure(lambda: wowml.linalg.svd(a_w)):.3f} ms")
    except (NotImplementedError, RuntimeError) as e:
        print(f"WowML linalg skipped: {e}")


if __name__ == "__main__":
    benchmark_linalg()
