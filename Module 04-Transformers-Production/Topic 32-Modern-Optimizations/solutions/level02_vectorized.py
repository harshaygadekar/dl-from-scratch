"""Level 02: blockwise score accumulation."""

import numpy as np


def blockwise_qk_scores(q, k, block=64):
    """q/k: (B, T, D) -> scores: (B, T, T)."""
    b, t, _ = q.shape
    out = np.zeros((b, t, t), dtype=q.dtype)
    for i in range(0, t, block):
        i2 = min(i + block, t)
        out[:, i:i2, :] = np.matmul(q[:, i:i2], np.swapaxes(k, -1, -2))
    return out
