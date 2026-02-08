"""Level 01: minimal seq2seq forward loop with teacher forcing."""

import numpy as np


def rnn_step(x_t, h_prev, w_xh, w_hh, b_h):
    return np.tanh(x_t @ w_xh + h_prev @ w_hh + b_h)


def one_hot(indices, vocab_size):
    return np.eye(vocab_size, dtype=np.float32)[indices]


def encode(src_tokens_one_hot, h0, w_xh, w_hh, b_h):
    """src_tokens_one_hot: (T_src, B, V)"""
    h_prev = h0
    for t in range(src_tokens_one_hot.shape[0]):
        h_prev = rnn_step(src_tokens_one_hot[t], h_prev, w_xh, w_hh, b_h)
    return h_prev


def decode_teacher_forcing(tgt_tokens_one_hot, h_init, w_xh, w_hh, b_h, w_out, b_out):
    """Return decoder logits over each time-step.

    tgt_tokens_one_hot: (T_tgt, B, V)
    returns logits: (T_tgt, B, V)
    """
    t_steps, batch, _ = tgt_tokens_one_hot.shape
    vocab = w_out.shape[1]
    logits = np.zeros((t_steps, batch, vocab), dtype=np.float32)

    h_prev = h_init
    for t in range(t_steps):
        h_prev = rnn_step(tgt_tokens_one_hot[t], h_prev, w_xh, w_hh, b_h)
        logits[t] = h_prev @ w_out + b_out

    return logits
