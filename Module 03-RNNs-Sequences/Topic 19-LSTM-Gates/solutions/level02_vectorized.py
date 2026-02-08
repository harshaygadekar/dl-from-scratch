"""Level 02: batch-first wrapper for LSTM."""

from level01_naive import lstm_forward


def lstm_forward_batch_first(x_batch_first, h0, c0, w_x, w_h, b):
    x_tbd = x_batch_first.transpose(1, 0, 2)
    h, c = lstm_forward(x_tbd, h0, c0, w_x, w_h, b)
    return h.transpose(1, 0, 2), c.transpose(1, 0, 2)
