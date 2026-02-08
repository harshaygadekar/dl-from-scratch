"""Level 03: chunked pooling for large batches."""

import numpy as np
from level02_vectorized import max_pool2d_vectorized


def max_pool2d_chunked(x, kernel_size=2, stride=2, chunk_size=8):
    outputs = []
    for start in range(0, x.shape[0], chunk_size):
        outputs.append(max_pool2d_vectorized(x[start:start + chunk_size], kernel_size=kernel_size, stride=stride))
    return np.concatenate(outputs, axis=0)
