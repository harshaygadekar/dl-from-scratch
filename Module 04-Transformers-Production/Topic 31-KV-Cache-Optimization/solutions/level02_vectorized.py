"""Level 02: cached attention for one query step."""

import numpy as np
from level01_naive import current_kv


def cached_attention_step(q_t, cache):
    """q_t: (B, H, 1, D), returns out: (B, H, 1, D)."""
    k, v = current_kv(cache)
    d = q_t.shape[-1]
    scores = np.matmul(q_t, np.swapaxes(k, -1, -2)) / np.sqrt(d)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / np.sum(w, axis=-1, keepdims=True)
    out = np.matmul(w, v)
    return out
