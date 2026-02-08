"""Level 01: Naive Im2Col implementation."""

import numpy as np


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


def im2col_naive(x, kernel_size, stride=1, padding=0):
    """Convert NCHW input to column matrix.

    Returns:
        cols: (N * H_out * W_out, C * K * K)
        out_h, out_w
    """
    n, c, h, w = x.shape
    k = kernel_size
    x_pad = pad_nchw(x, padding)
    h_out = compute_output_size(h, k, stride, padding)
    w_out = compute_output_size(w, k, stride, padding)

    cols = np.zeros((n * h_out * w_out, c * k * k), dtype=x.dtype)
    row = 0

    for b in range(n):
        for i in range(h_out):
            for j in range(w_out):
                patch = x_pad[b, :, i * stride:i * stride + k, j * stride:j * stride + k]
                cols[row] = patch.reshape(-1)
                row += 1

    return cols, h_out, w_out


def conv2d_im2col_naive(x, weight, bias=None, stride=1, padding=0):
    """Conv2D via im2col + matmul.

    x: (N, C_in, H, W)
    weight: (C_out, C_in, K, K)
    """
    n, c_in, _, _ = x.shape
    c_out, c_w, k, _ = weight.shape
    if c_in != c_w:
        raise ValueError("input channels and weight channels must match")

    cols, h_out, w_out = im2col_naive(x, kernel_size=k, stride=stride, padding=padding)
    w_col = weight.reshape(c_out, -1).T
    out = cols @ w_col

    if bias is not None:
        out += bias.reshape(1, -1)

    out = out.reshape(n, h_out, w_out, c_out).transpose(0, 3, 1, 2)
    return out


if __name__ == "__main__":
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    y = conv2d_im2col_naive(x, w, stride=1, padding=1)
    print("Output shape:", y.shape)
