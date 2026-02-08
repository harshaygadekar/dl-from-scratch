"""Level 01: Bahdanau attention score/context computation."""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def bahdanau_scores(query, keys, w_q, w_k, v_a, b_a):
    """Compute additive attention scores.

    query: (B, H_dec)
    keys: (T, B, H_enc)
    w_q: (H_dec, A)
    w_k: (H_enc, A)
    v_a: (A,)
    b_a: (A,)

    returns: (B, T)
    """
    t_steps, batch, _ = keys.shape
    scores = np.zeros((batch, t_steps), dtype=np.float32)

    q_proj = query @ w_q
    for t in range(t_steps):
        k_proj = keys[t] @ w_k
        energy = np.tanh(q_proj + k_proj + b_a)
        scores[:, t] = energy @ v_a

    return scores


def bahdanau_context(query, keys, values, w_q, w_k, v_a, b_a, mask=None):
    """Return context vector and attention weights.

    values: (T, B, H_val)
    mask: (B, T) with 1 for valid tokens, 0 for padded.
    """
    scores = bahdanau_scores(query, keys, w_q, w_k, v_a, b_a)

    if mask is not None:
        scores = np.where(mask > 0, scores, -1e9)

    weights = softmax(scores, axis=1)

    values_btv = values.transpose(1, 0, 2)
    context = np.einsum("bt,bth->bh", weights, values_btv)
    return context, weights
