"""Level 02: batch-first wrappers for masked bidirectional RNN."""

from level01_naive import bidirectional_rnn_forward, apply_mask_batch_first


def bidirectional_rnn_batch_first(x_batch_first, *args):
    """x_batch_first: (B, T, D) -> (B, T, 2H)."""
    x_tbd = x_batch_first.transpose(1, 0, 2)
    out_tbd = bidirectional_rnn_forward(x_tbd, *args)
    return out_tbd.transpose(1, 0, 2)
