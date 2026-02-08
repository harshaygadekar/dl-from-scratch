"""Level 03: checkpoint-friendly residual execution placeholder."""

from level02_vectorized import residual_stack


def residual_stack_chunked(x, blocks, chunk_size=8):
    outputs = []
    for start in range(0, x.shape[0], chunk_size):
        outputs.append(residual_stack(x[start:start + chunk_size], blocks))
    import numpy as np

    return np.concatenate(outputs, axis=0)
