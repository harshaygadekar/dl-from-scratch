"""Level 02: vectorized residual stack helpers."""

import numpy as np
from level01_naive import residual_block_forward


def residual_stack(x, blocks):
    """Apply a sequence of residual block parameter dicts."""
    out = x
    for params in blocks:
        out = residual_block_forward(out, **params)
    return out
