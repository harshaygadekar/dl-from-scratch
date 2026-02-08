"""Level 03: stream hidden states instead of materializing full sequence."""

from level01_naive import rnn_step


def rnn_hidden_generator(x, h0, w_xh, w_hh, b_h):
    """Yield hidden state at each time-step for memory-aware pipelines."""
    h_prev = h0
    for t in range(x.shape[0]):
        h_prev = rnn_step(x[t], h_prev, w_xh, w_hh, b_h)
        yield h_prev
