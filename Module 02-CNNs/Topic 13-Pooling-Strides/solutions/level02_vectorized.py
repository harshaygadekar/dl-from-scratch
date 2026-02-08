"""Level 02: vectorized pooling wrappers."""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def _windows_nchw(x, kernel_size, stride):
    n, c, h, w = x.shape
    h_out = (h - kernel_size) // stride + 1
    w_out = (w - kernel_size) // stride + 1
    windows = as_strided(
        x,
        shape=(n, c, h_out, w_out, kernel_size, kernel_size),
        strides=(
            x.strides[0],
            x.strides[1],
            x.strides[2] * stride,
            x.strides[3] * stride,
            x.strides[2],
            x.strides[3],
        ),
    )
    return windows


def max_pool2d_vectorized(x, kernel_size=2, stride=2):
    windows = _windows_nchw(x, kernel_size, stride)
    return windows.max(axis=(-1, -2))


def avg_pool2d_vectorized(x, kernel_size=2, stride=2):
    windows = _windows_nchw(x, kernel_size, stride)
    n, c, h_out, w_out, _, _ = windows.shape
    out = np.empty((n, c, h_out, w_out), dtype=x.dtype)

    # Compute each spatial window mean independently to mirror level01 numerics.
    for i in range(h_out):
        for j in range(w_out):
            out[:, :, i, j] = windows[:, :, i, j].mean(axis=(-1, -2))

    return out
