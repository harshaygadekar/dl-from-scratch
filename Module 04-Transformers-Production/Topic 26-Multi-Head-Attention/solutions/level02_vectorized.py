"""Level 02: vectorized per-head attention core."""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def multi_head_attention_vectorized(qh, kh, vh, mask=None):
    """qh/kh/vh: (B, H, T, D_head) -> out: (B, H, T, D_head)"""
    d = qh.shape[-1]
    scores = np.matmul(qh, np.swapaxes(kh, -1, -2)) / np.sqrt(d)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        scores = np.where(mask > 0, scores, -1e9)

    weights = softmax(scores, axis=-1)
    return np.matmul(weights, vh)
