"""Level 03: in-place friendly normalization wrappers."""

from level01_naive import instance_norm_nchw


def instance_norm_inplace(x, eps=1e-5):
    x_hat = instance_norm_nchw(x, eps=eps)
    x[...] = x_hat
    return x
