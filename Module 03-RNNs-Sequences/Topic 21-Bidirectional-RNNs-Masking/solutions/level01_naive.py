"""Level 01: bidirectional RNN and mask utilities."""

import numpy as np


def rnn_step(x_t, h_prev, w_xh, w_hh, b_h):
    return np.tanh(x_t @ w_xh + h_prev @ w_hh + b_h)


def rnn_forward(x, h0, w_xh, w_hh, b_h):
    """Unrolled RNN forward over time.

    x: (T, B, D)
    returns: (T, B, H)
    """
    t_steps, batch, _ = x.shape
    h = np.zeros((t_steps, batch, h0.shape[1]), dtype=x.dtype)
    h_prev = h0

    for t in range(t_steps):
        h_prev = rnn_step(x[t], h_prev, w_xh, w_hh, b_h)
        h[t] = h_prev

    return h


def sequence_mask(lengths, max_len=None):
    lengths = np.asarray(lengths)
    if max_len is None:
        max_len = int(lengths.max())
    steps = np.arange(max_len)[None, :]
    return (steps < lengths[:, None]).astype(np.float32)


def apply_mask_batch_first(x, mask):
    """x: (B, T, C), mask: (B, T)."""
    return x * mask[:, :, None]


def bidirectional_rnn_forward(x, h0_f, h0_b, w_xh_f, w_hh_f, b_f, w_xh_b, w_hh_b, b_b):
    """x: (T, B, D), output: (T, B, 2H)."""
    h_f = rnn_forward(x, h0_f, w_xh_f, w_hh_f, b_f)
    h_b_rev = rnn_forward(x[::-1], h0_b, w_xh_b, w_hh_b, b_b)
    h_b = h_b_rev[::-1]
    return np.concatenate([h_f, h_b], axis=2)
