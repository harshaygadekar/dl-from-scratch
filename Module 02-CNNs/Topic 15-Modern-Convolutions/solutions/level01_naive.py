"""Level 01: naive modern convolution variants."""

import numpy as np


def _pad_nchw(x, padding):
    if padding == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")


def depthwise_conv2d(x, weight, stride=1, padding=0):
    """Depthwise conv.

    x: (N, C, H, W)
    weight: (C, K, K)
    """
    n, c, h, w = x.shape
    c_w, k, _ = weight.shape
    if c != c_w:
        raise ValueError("depthwise weight channels must match input channels")

    x_pad = _pad_nchw(x, padding)
    h_out = (h + 2 * padding - k) // stride + 1
    w_out = (w + 2 * padding - k) // stride + 1
    out = np.zeros((n, c, h_out, w_out), dtype=x.dtype)

    for b in range(n):
        for ch in range(c):
            for i in range(h_out):
                for j in range(w_out):
                    h0 = i * stride
                    w0 = j * stride
                    patch = x_pad[b, ch, h0:h0 + k, w0:w0 + k]
                    out[b, ch, i, j] = np.sum(patch * weight[ch])

    return out


def pointwise_conv2d(x, weight, bias=None):
    """Pointwise 1x1 convolution.

    x: (N, C_in, H, W)
    weight: (C_out, C_in)
    """
    n, c_in, h, w = x.shape
    c_out, c_w = weight.shape
    if c_in != c_w:
        raise ValueError("pointwise channels mismatch")

    x_nhwc = x.transpose(0, 2, 3, 1)
    y_nhwc = np.tensordot(x_nhwc, weight.T, axes=([3], [0]))
    if bias is not None:
        y_nhwc += bias.reshape(1, 1, 1, -1)
    return y_nhwc.transpose(0, 3, 1, 2)


def dilated_conv2d(x, weight, dilation=2, stride=1, padding=0):
    """Single-kernel-size dilated convolution.

    x: (N, C_in, H, W)
    weight: (C_out, C_in, K, K)
    """
    n, c_in, h, w = x.shape
    c_out, c_w, k, _ = weight.shape
    if c_in != c_w:
        raise ValueError("channel mismatch")

    x_pad = _pad_nchw(x, padding)
    effective_k = (k - 1) * dilation + 1
    h_out = (h + 2 * padding - effective_k) // stride + 1
    w_out = (w + 2 * padding - effective_k) // stride + 1

    out = np.zeros((n, c_out, h_out, w_out), dtype=x.dtype)

    for b in range(n):
        for co in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    h0 = i * stride
                    w0 = j * stride
                    acc = 0.0
                    for ci in range(c_in):
                        for kh in range(k):
                            for kw in range(k):
                                h_idx = h0 + kh * dilation
                                w_idx = w0 + kw * dilation
                                acc += x_pad[b, ci, h_idx, w_idx] * weight[co, ci, kh, kw]
                    out[b, co, i, j] = acc

    return out
