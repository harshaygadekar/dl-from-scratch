"""Level 01: normalization primitives for NCHW tensors."""

import numpy as np


def layer_norm_nchw(x, eps=1e-5):
    """Normalize over (C, H, W) for each sample."""
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    var = x.var(axis=(1, 2, 3), keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat


def instance_norm_nchw(x, eps=1e-5):
    """Normalize each (N, C) map over (H, W)."""
    mean = x.mean(axis=(2, 3), keepdims=True)
    var = x.var(axis=(2, 3), keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def group_norm_nchw(x, num_groups, eps=1e-5):
    """Normalize channel groups over (group_channels, H, W)."""
    n, c, h, w = x.shape
    if c % num_groups != 0:
        raise ValueError("channels must be divisible by num_groups")

    g = num_groups
    x_grouped = x.reshape(n, g, c // g, h, w)
    mean = x_grouped.mean(axis=(2, 3, 4), keepdims=True)
    var = x_grouped.var(axis=(2, 3, 4), keepdims=True)
    x_hat = (x_grouped - mean) / np.sqrt(var + eps)
    return x_hat.reshape(n, c, h, w)
