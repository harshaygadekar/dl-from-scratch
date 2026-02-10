#!/usr/bin/env python3
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "solutions"))
from level01_naive import greedy_next
from level02_vectorized import speculative_accept_reject


if __name__ == "__main__":
    np.random.seed(0)
    vocab = 5000
    logits = np.random.randn(vocab).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(2000):
        _ = greedy_next(logits)
    t_serial = time.perf_counter() - t0

    draft = np.argsort(logits)[::-1][:8]
    t0 = time.perf_counter()
    for _ in range(2000):
        _ = speculative_accept_reject(logits, draft)
    t_spec = time.perf_counter() - t0

    print(f"baseline_seconds={t_serial:.4f}")
    print(f"optimized_seconds={t_spec:.4f}")
    print(f"speedup_proxy={t_serial / max(t_spec, 1e-12):.2f}x")
