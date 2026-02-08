"""Level 01: minimal Transformer encoder block."""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, :, :]
        scores = np.where(mask > 0, scores, -1e9)

    weights = softmax(scores, axis=-1)
    out = np.matmul(weights, v)
    return out, weights


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def ffn(x, w1, b1, w2, b2):
    hidden = np.maximum(0, x @ w1 + b1)
    return hidden @ w2 + b2


def encoder_block_forward(x, w_q, w_k, w_v, w_o, w1, b1, w2, b2):
    attn_out, _ = scaled_dot_product_attention(x @ w_q, x @ w_k, x @ w_v)
    x = layer_norm(x + attn_out @ w_o)
    x = layer_norm(x + ffn(x, w1, b1, w2, b2))
    return x
