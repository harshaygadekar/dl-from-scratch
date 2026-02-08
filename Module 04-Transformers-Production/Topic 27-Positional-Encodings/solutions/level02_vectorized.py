"""Level 02: simple RoPE application utility."""

import numpy as np


def apply_rope(x):
    """Apply RoPE-style rotation to last dim pairs.

    x: (B, T, D), D must be even.
    """
    b, t, d = x.shape
    if d % 2 != 0:
        raise ValueError("RoPE requires even embedding dimension")

    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]

    pos = np.arange(t, dtype=np.float32)[None, :, None]
    freq = 1.0 / np.power(10000.0, np.arange(half, dtype=np.float32)[None, None, :] / half)
    theta = pos * freq

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    y1 = x1 * cos_t - x2 * sin_t
    y2 = x1 * sin_t + x2 * cos_t
    return np.concatenate([y1, y2], axis=-1)
