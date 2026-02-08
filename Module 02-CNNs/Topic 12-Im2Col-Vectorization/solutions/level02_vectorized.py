"""Level 02: Vectorized Im2Col implementation with stride views."""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def compute_output_size(size, kernel, stride, padding):
    return (size + 2 * padding - kernel) // stride + 1


def pad_nchw(x, padding):
    if padding == 0:
        return x
    return np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )


def im2col_vectorized(x, kernel_size, stride=1, padding=0):
    n, c, h, w = x.shape
    k = kernel_size
    x_pad = pad_nchw(x, padding)
    h_out = compute_output_size(h, k, stride, padding)
    w_out = compute_output_size(w, k, stride, padding)

    windows = as_strided(
        x_pad,
        shape=(n, c, h_out, w_out, k, k),
        strides=(
            x_pad.strides[0],
            x_pad.strides[1],
            x_pad.strides[2] * stride,
            x_pad.strides[3] * stride,
            x_pad.strides[2],
            x_pad.strides[3],
        ),
    )

    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h_out * w_out, c * k * k)
    return cols, h_out, w_out


def conv2d_im2col_vectorized(x, weight, bias=None, stride=1, padding=0):
    n, c_in, _, _ = x.shape
    c_out, c_w, k, _ = weight.shape
    if c_in != c_w:
        raise ValueError("input channels and weight channels must match")

    cols, h_out, w_out = im2col_vectorized(x, kernel_size=k, stride=stride, padding=padding)
    out = cols @ weight.reshape(c_out, -1).T

    if bias is not None:
        out += bias.reshape(1, -1)

    return out.reshape(n, h_out, w_out, c_out).transpose(0, 3, 1, 2)


if __name__ == "__main__":
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    y = conv2d_im2col_vectorized(x, w, stride=1, padding=1)
    print("Output shape:", y.shape)
