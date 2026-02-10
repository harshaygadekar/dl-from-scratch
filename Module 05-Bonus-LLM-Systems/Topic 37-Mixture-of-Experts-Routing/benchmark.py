#!/usr/bin/env python3
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "solutions"))
from level01_naive import top1_route
from level02_vectorized import route_with_capacity


if __name__ == "__main__":
    np.random.seed(1)
    logits = np.random.randn(100_000, 16).astype(np.float32)

    t0 = time.perf_counter()
    top1 = top1_route(logits)
    t_route = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, counts = route_with_capacity(top1, num_experts=16, capacity=8000)
    t_cap = time.perf_counter() - t0

    print(f"top1_seconds={t_route:.4f}")
    print(f"capacity_seconds={t_cap:.4f}")
    print(f"experts_used={(counts > 0).sum()}")
