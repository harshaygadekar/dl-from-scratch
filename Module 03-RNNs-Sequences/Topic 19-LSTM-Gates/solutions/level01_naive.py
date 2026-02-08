"""Level 01: LSTM cell with explicit gate computations."""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_step(x_t, h_prev, c_prev, w_x, w_h, b):
    """Single LSTM step.

    w_x: (D, 4H), w_h: (H, 4H), b: (4H,)
    """
    z = x_t @ w_x + h_prev @ w_h + b
    h_size = h_prev.shape[1]

    i = sigmoid(z[:, 0:h_size])
    f = sigmoid(z[:, h_size:2 * h_size])
    g = np.tanh(z[:, 2 * h_size:3 * h_size])
    o = sigmoid(z[:, 3 * h_size:4 * h_size])

    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)
    return h_t, c_t


def lstm_forward(x, h0, c0, w_x, w_h, b):
    t_steps, batch, _ = x.shape
    h_size = h0.shape[1]
    h_out = np.zeros((t_steps, batch, h_size), dtype=x.dtype)
    c_out = np.zeros((t_steps, batch, h_size), dtype=x.dtype)

    h_prev, c_prev = h0, c0
    for t in range(t_steps):
        h_prev, c_prev = lstm_step(x[t], h_prev, c_prev, w_x, w_h, b)
        h_out[t] = h_prev
        c_out[t] = c_prev

    return h_out, c_out
