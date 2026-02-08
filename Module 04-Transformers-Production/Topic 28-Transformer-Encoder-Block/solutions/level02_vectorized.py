"""Level 02: stacked encoder application."""

from level01_naive import encoder_block_forward


def encoder_stack(x, blocks):
    for block in blocks:
        x = encoder_block_forward(x, **block)
    return x
