"""Level 02: batched vanilla RNN wrappers."""

import numpy as np
from level01_naive import rnn_forward


def rnn_forward_batch_first(x_batch_first, h0, w_xh, w_hh, b_h):
    """x_batch_first: (B, T, D) -> output (B, T, H)."""
    x_tbd = x_batch_first.transpose(1, 0, 2)
    h_tbd = rnn_forward(x_tbd, h0, w_xh, w_hh, b_h)
    return h_tbd.transpose(1, 0, 2)
