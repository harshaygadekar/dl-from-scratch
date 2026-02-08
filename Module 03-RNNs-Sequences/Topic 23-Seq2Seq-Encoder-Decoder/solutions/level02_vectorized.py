"""Level 02: greedy decoder utility."""

import numpy as np
from level01_naive import rnn_step, one_hot


def decode_greedy(start_token, h_init, steps, vocab_size, w_xh, w_hh, b_h, w_out, b_out):
    """Greedy autoregressive decode in token id space."""
    batch = h_init.shape[0]
    tokens = np.full((batch,), start_token, dtype=np.int64)
    h_prev = h_init
    outputs = []

    for _ in range(steps):
        x_t = one_hot(tokens, vocab_size)
        h_prev = rnn_step(x_t, h_prev, w_xh, w_hh, b_h)
        logits = h_prev @ w_out + b_out
        tokens = np.argmax(logits, axis=1)
        outputs.append(tokens.copy())

    return np.stack(outputs, axis=0)
