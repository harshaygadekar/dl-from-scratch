"""Level 03: chunked score computation utility."""

import numpy as np


def chunked_attention_scores(q, k, chunk_size=64):
    """Compute q @ k^T in query chunks.

    q, k: (B, T, D)
    returns scores: (B, T, T)
    """
    b, t, _ = q.shape
    scores = np.zeros((b, t, t), dtype=q.dtype)

    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        q_chunk = q[:, start:end, :]
        scores[:, start:end, :] = np.matmul(q_chunk, np.swapaxes(k, -1, -2))

    return scores
