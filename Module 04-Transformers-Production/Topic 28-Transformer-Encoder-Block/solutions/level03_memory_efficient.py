"""Level 03: chunked FFN for memory control."""

import numpy as np


def chunked_ffn(x, w1, b1, w2, b2, chunk_size=64):
    b, t, d = x.shape
    out = np.zeros((b, t, d), dtype=x.dtype)
    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        h = np.maximum(0, x[:, start:end] @ w1 + b1)
        out[:, start:end] = h @ w2 + b2
    return out
