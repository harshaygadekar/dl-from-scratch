"""Level 02: affine wrappers for normalization outputs."""

import numpy as np
from level01_naive import layer_norm_nchw, instance_norm_nchw, group_norm_nchw


def apply_affine(x_hat, gamma, beta):
    return x_hat * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)


def group_norm_affine(x, num_groups, gamma, beta, eps=1e-5):
    x_hat = group_norm_nchw(x, num_groups=num_groups, eps=eps)
    return apply_affine(x_hat, gamma=gamma, beta=beta)
