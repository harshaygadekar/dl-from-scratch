"""Level 01: GRU cell implementation."""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_step(x_t, h_prev, w_x, w_h, b):
    """Single GRU step.

    w_x: (D, 3H), w_h: (H, 3H), b: (3H,)
    Gates: update z, reset r, candidate n
    """
    h = h_prev.shape[1]

    z = sigmoid(x_t @ w_x[:, :h] + h_prev @ w_h[:, :h] + b[:h])
    r = sigmoid(x_t @ w_x[:, h:2 * h] + h_prev @ w_h[:, h:2 * h] + b[h:2 * h])

    n_pre = x_t @ w_x[:, 2 * h:3 * h] + (r * h_prev) @ w_h[:, 2 * h:3 * h] + b[2 * h:3 * h]
    n = np.tanh(n_pre)

    h_t = (1.0 - z) * n + z * h_prev
    return h_t


def gru_forward(x, h0, w_x, w_h, b):
    t_steps, batch, _ = x.shape
    h_size = h0.shape[1]
    out = np.zeros((t_steps, batch, h_size), dtype=x.dtype)

    h_prev = h0
    for t in range(t_steps):
        h_prev = gru_step(x[t], h_prev, w_x, w_h, b)
        out[t] = h_prev

    return out
