import numpy as np


def chunked_selective_scan(a, b, x, chunk=128):
    out = np.zeros_like(x, dtype=np.float32)
    prev = 0.0
    n = len(x)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        for i in range(start, end):
            prev = a[i] * prev + b[i] * x[i]
            out[i] = prev
    return out
