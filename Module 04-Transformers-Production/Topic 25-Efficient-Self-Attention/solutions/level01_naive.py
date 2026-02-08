"""Level 01: scaled dot-product attention with optional mask."""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def causal_mask(seq_len):
    """Lower-triangular causal mask with 1 for allowed positions."""
    return np.tril(np.ones((seq_len, seq_len), dtype=np.float32))


def scaled_dot_product_attention(q, k, v, mask=None):
    """Compute attention.

    q, k, v: (B, T, D)
    mask: (B, T, T) or (T, T), 1 for keep and 0 for block
    returns: output (B, T, D), weights (B, T, T)
    """
    d = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, :, :]
        scores = np.where(mask > 0, scores, -1e9)

    weights = softmax(scores, axis=-1)
    out = np.matmul(weights, v)
    return out, weights
