"""Level 03: memory-aware wrappers around vectorized Im2Col."""

import numpy as np
from level02_vectorized import conv2d_im2col_vectorized


def conv2d_im2col_chunked(x, weight, bias=None, stride=1, padding=0, chunk_size=8):
    outputs = []
    for start in range(0, x.shape[0], chunk_size):
        x_chunk = x[start:start + chunk_size]
        y_chunk = conv2d_im2col_vectorized(x_chunk, weight, bias=bias, stride=stride, padding=padding)
        outputs.append(y_chunk)
    return np.concatenate(outputs, axis=0)


if __name__ == "__main__":
    x = np.random.randn(16, 3, 16, 16).astype(np.float32)
    w = np.random.randn(8, 3, 3, 3).astype(np.float32)
    y = conv2d_im2col_chunked(x, w, stride=1, padding=1, chunk_size=4)
    print("Output shape:", y.shape)
