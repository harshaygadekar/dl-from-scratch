#!/usr/bin/env python3
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "solutions"))
from level01_naive import apply_lora
from level03_memory_efficient import apply_lora_inplace


def bench(fn, loops=30):
    w = np.random.randn(256, 256).astype(np.float32)
    a = np.random.randn(256, 8).astype(np.float32)
    b = np.random.randn(8, 256).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(loops):
        _ = fn(w.copy(), a, b, alpha=1.0)
    return time.perf_counter() - t0


if __name__ == "__main__":
    t_base = bench(apply_lora)
    t_opt = bench(apply_lora_inplace)
    print(f"baseline_seconds={t_base:.4f}")
    print(f"optimized_seconds={t_opt:.4f}")
    print(f"speedup={t_base / max(t_opt, 1e-12):.2f}x")
