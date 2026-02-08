"""Level 02: autoregressive step helper."""

import numpy as np
from level01_naive import decoder_block_forward


def decoder_autoregressive_step(prefix, enc_out, params):
    """Compute block output and return last timestep representation."""
    out = decoder_block_forward(prefix, enc_out, params)
    return out[:, -1, :]
