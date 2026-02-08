"""Level 01: Vanilla RNN with explicit time-step loop."""

import numpy as np


def tanh(x):
    return np.tanh(x)


def rnn_step(x_t, h_prev, w_xh, w_hh, b_h):
    """Single RNN step.

    x_t: (B, D)
    h_prev: (B, H)
    w_xh: (D, H)
    w_hh: (H, H)
    b_h: (H,)
    """
    return tanh(x_t @ w_xh + h_prev @ w_hh + b_h)


def rnn_forward(x, h0, w_xh, w_hh, b_h):
    """Unrolled RNN forward.

    x: (T, B, D)
    returns h: (T, B, H)
    """
    t_steps, batch, _ = x.shape
    h = np.zeros((t_steps, batch, h0.shape[1]), dtype=x.dtype)
    h_prev = h0

    for t in range(t_steps):
        h_t = rnn_step(x[t], h_prev, w_xh, w_hh, b_h)
        h[t] = h_t
        h_prev = h_t

    return h


def clip_grad_norm(grad, max_norm=1.0):
    norm = float(np.linalg.norm(grad))
    if norm <= max_norm or norm == 0.0:
        return grad
    return grad * (max_norm / norm)
