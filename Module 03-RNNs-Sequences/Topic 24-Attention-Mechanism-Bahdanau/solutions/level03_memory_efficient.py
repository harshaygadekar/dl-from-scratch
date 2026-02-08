"""Level 03: top-k attention pruning utility."""

import numpy as np


def topk_attention_mask(scores, k):
    """Keep only top-k scores per row and mask others."""
    if k >= scores.shape[1]:
        return np.ones_like(scores, dtype=np.float32)

    idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    mask = np.zeros_like(scores, dtype=np.float32)
    rows = np.arange(scores.shape[0])[:, None]
    mask[rows, idx] = 1.0
    return mask
