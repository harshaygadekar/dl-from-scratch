"""Level 03: streaming GRU hidden states."""

from level01_naive import gru_step


def gru_generator(x, h0, w_x, w_h, b):
    h_prev = h0
    for t in range(x.shape[0]):
        h_prev = gru_step(x[t], h_prev, w_x, w_h, b)
        yield h_prev
