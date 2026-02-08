"""Level 01: minimal residual block primitives."""

import numpy as np


def relu(x):
    return np.maximum(0, x)


def conv1x1(x, weight, bias=None):
    """1x1 convolution over NCHW input.

    x: (N, C_in, H, W)
    weight: (C_out, C_in)
    """
    n, c_in, h, w = x.shape
    c_out, c_w = weight.shape
    if c_in != c_w:
        raise ValueError("channel mismatch")

    x_nhwc = x.transpose(0, 2, 3, 1)
    y_nhwc = np.tensordot(x_nhwc, weight.T, axes=([3], [0]))
    if bias is not None:
        y_nhwc += bias.reshape(1, 1, 1, -1)
    return y_nhwc.transpose(0, 3, 1, 2)


def residual_block_forward(x, w1, w2, b1=None, b2=None, proj_w=None, proj_b=None):
    """Simple residual block: y = ReLU(conv2(ReLU(conv1(x))) + shortcut)."""
    out = conv1x1(x, w1, b1)
    out = relu(out)
    out = conv1x1(out, w2, b2)

    if proj_w is not None:
        shortcut = conv1x1(x, proj_w, proj_b)
    else:
        shortcut = x

    return relu(out + shortcut)
