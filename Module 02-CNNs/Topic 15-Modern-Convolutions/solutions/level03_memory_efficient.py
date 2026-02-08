"""Level 03: chunked depthwise-separable wrapper."""

import numpy as np
from level02_vectorized import depthwise_separable_conv2d


def depthwise_separable_chunked(x, depthwise_w, pointwise_w, pointwise_b=None, stride=1, padding=0, chunk_size=8):
    outputs = []
    for start in range(0, x.shape[0], chunk_size):
        y = depthwise_separable_conv2d(
            x[start:start + chunk_size],
            depthwise_w,
            pointwise_w,
            pointwise_b=pointwise_b,
            stride=stride,
            padding=padding,
        )
        outputs.append(y)
    return np.concatenate(outputs, axis=0)
