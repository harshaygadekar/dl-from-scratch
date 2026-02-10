#!/usr/bin/env python3
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "solutions"))
from level01_naive import selective_scan_naive
from level03_memory_efficient import chunked_selective_scan


if __name__ == "__main__":
    np.random.seed(2)
    n = 200_000
    a = np.random.uniform(0.8, 0.99, size=n).astype(np.float32)
    b = np.random.uniform(0.01, 0.2, size=n).astype(np.float32)
    x = np.random.randn(n).astype(np.float32)

    t0 = time.perf_counter()
    _ = selective_scan_naive(a, b, x)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = chunked_selective_scan(a, b, x, chunk=2048)
    t_chunk = time.perf_counter() - t0

    print(f"baseline_seconds={t_naive:.4f}")
    print(f"optimized_seconds={t_chunk:.4f}")
    print(f"speedup={t_naive / max(t_chunk, 1e-12):.2f}x")
