"""Level 01: naive pooling implementations."""

import numpy as np


def compute_output_size(size, kernel, stride, padding):
    return (size + 2 * padding - kernel) // stride + 1


def pad_nchw(x, padding):
    if padding == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")


def max_pool2d_forward(x, kernel_size=2, stride=2, padding=0):
    x_pad = pad_nchw(x, padding)
    n, c, h, w = x_pad.shape
    h_out = compute_output_size(h, kernel_size, stride, 0)
    w_out = compute_output_size(w, kernel_size, stride, 0)

    out = np.zeros((n, c, h_out, w_out), dtype=x.dtype)
    mask = np.zeros_like(x_pad, dtype=np.uint8)

    for b in range(n):
        for ch in range(c):
            for i in range(h_out):
                for j in range(w_out):
                    h0 = i * stride
                    w0 = j * stride
                    window = x_pad[b, ch, h0:h0 + kernel_size, w0:w0 + kernel_size]
                    idx = np.unravel_index(np.argmax(window), window.shape)
                    out[b, ch, i, j] = window[idx]
                    mask[b, ch, h0 + idx[0], w0 + idx[1]] = 1

    cache = (x.shape, kernel_size, stride, padding, mask)
    return out, cache


def max_pool2d_backward(grad_out, cache):
    input_shape, kernel_size, stride, padding, mask = cache
    n, c, h, w = mask.shape
    _, _, h_out, w_out = grad_out.shape

    grad_in_pad = np.zeros_like(mask, dtype=grad_out.dtype)
    for b in range(n):
        for ch in range(c):
            for i in range(h_out):
                for j in range(w_out):
                    h0 = i * stride
                    w0 = j * stride
                    region = mask[b, ch, h0:h0 + kernel_size, w0:w0 + kernel_size]
                    grad_in_pad[b, ch, h0:h0 + kernel_size, w0:w0 + kernel_size] += region * grad_out[b, ch, i, j]

    if padding == 0:
        return grad_in_pad

    _, _, h_in, w_in = input_shape
    return grad_in_pad[:, :, padding:padding + h_in, padding:padding + w_in]


def avg_pool2d_forward(x, kernel_size=2, stride=2, padding=0):
    x_pad = pad_nchw(x, padding)
    n, c, h, w = x_pad.shape
    h_out = compute_output_size(h, kernel_size, stride, 0)
    w_out = compute_output_size(w, kernel_size, stride, 0)

    out = np.zeros((n, c, h_out, w_out), dtype=x.dtype)
    for b in range(n):
        for ch in range(c):
            for i in range(h_out):
                for j in range(w_out):
                    h0 = i * stride
                    w0 = j * stride
                    window = x_pad[b, ch, h0:h0 + kernel_size, w0:w0 + kernel_size]
                    out[b, ch, i, j] = window.mean()
    return out
