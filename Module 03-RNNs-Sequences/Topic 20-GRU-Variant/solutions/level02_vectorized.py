"""Level 02: batch-first GRU wrapper."""

from level01_naive import gru_forward


def gru_forward_batch_first(x_batch_first, h0, w_x, w_h, b):
    x_tbd = x_batch_first.transpose(1, 0, 2)
    h = gru_forward(x_tbd, h0, w_x, w_h, b)
    return h.transpose(1, 0, 2)
