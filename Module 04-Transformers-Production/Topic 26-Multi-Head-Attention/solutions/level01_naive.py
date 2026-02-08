"""Level 01: basic multi-head attention shape logic."""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    """q/k/v: (B, T, D), mask: (T,T) or (B,T,T)."""
    d = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, :, :]
        scores = np.where(mask > 0, scores, -1e9)

    weights = softmax(scores, axis=-1)
    out = np.matmul(weights, v)
    return out, weights


def split_heads(x, num_heads):
    """x: (B, T, D) -> (B, H, T, D_head)"""
    b, t, d = x.shape
    if d % num_heads != 0:
        raise ValueError("model dimension must be divisible by num_heads")
    d_head = d // num_heads
    return x.reshape(b, t, num_heads, d_head).transpose(0, 2, 1, 3)


def combine_heads(x):
    """x: (B, H, T, D_head) -> (B, T, D)"""
    b, h, t, d_head = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * d_head)


def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, mask=None):
    """Self-attention with learned projections.

    x: (B, T, D)
    w_q/w_k/w_v/w_o: (D, D)
    """
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    qh = split_heads(q, num_heads)
    kh = split_heads(k, num_heads)
    vh = split_heads(v, num_heads)

    b, h, t, d_head = qh.shape
    out_heads = np.zeros_like(qh)

    for head in range(h):
        out_h, _ = scaled_dot_product_attention(
            qh[:, head], kh[:, head], vh[:, head], mask=mask
        )
        out_heads[:, head] = out_h

    out = combine_heads(out_heads)
    return out @ w_o
