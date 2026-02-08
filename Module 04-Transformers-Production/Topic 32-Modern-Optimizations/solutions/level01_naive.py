"""Level 01: optimization utility primitives."""

import numpy as np


def split_segments(length, segment_size):
    segments = []
    for start in range(0, length, segment_size):
        end = min(start + segment_size, length)
        segments.append((start, end))
    return segments


def checkpoint_plan(num_layers, every_n_layers=2):
    """Return which layers to checkpoint."""
    return [i for i in range(num_layers) if i % every_n_layers == 0]


def estimate_attention_flops(seq_len, d_model):
    """Rough attention FLOPs scaling O(T^2 * D)."""
    return 2 * (seq_len ** 2) * d_model
