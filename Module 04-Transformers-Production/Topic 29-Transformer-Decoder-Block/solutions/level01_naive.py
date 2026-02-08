"""Level 01: minimal decoder block with masked self-attn + cross-attn."""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len), dtype=np.float32))


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


def decoder_block_forward(x, enc_out, params):
    """x: (B, T_dec, D), enc_out: (B, T_enc, D)"""
    t_dec = x.shape[1]
    m = causal_mask(t_dec)

    self_out, _ = scaled_dot_product_attention(
        x @ params["w_qs"], x @ params["w_ks"], x @ params["w_vs"], mask=m
    )
    x = layer_norm(x + self_out @ params["w_os"])

    cross_out, _ = scaled_dot_product_attention(
        x @ params["w_qc"], enc_out @ params["w_kc"], enc_out @ params["w_vc"]
    )
    x = layer_norm(x + cross_out @ params["w_oc"])

    hidden = np.maximum(0, x @ params["w1"] + params["b1"])
    x = layer_norm(x + hidden @ params["w2"] + params["b2"])
    return x
