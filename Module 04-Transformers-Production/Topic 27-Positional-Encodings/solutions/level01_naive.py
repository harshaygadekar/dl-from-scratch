"""Level 01: sinusoidal positional encoding."""

import numpy as np


def sinusoidal_positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, None]
    dims = np.arange(d_model)[None, :]

    angle_rates = 1.0 / np.power(10000.0, (2 * (dims // 2)) / d_model)
    angles = positions * angle_rates

    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def add_positional_encoding(x, pe):
    """x: (B, T, D), pe: (T, D)."""
    return x + pe[None, :, :]
