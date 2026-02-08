"""Level 03: grouped-head processing utility."""

import numpy as np
from level02_vectorized import multi_head_attention_vectorized


def mha_head_chunks(qh, kh, vh, head_chunk=2, mask=None):
    b, h, t, d = qh.shape
    out = np.zeros_like(qh)

    for start in range(0, h, head_chunk):
        end = min(start + head_chunk, h)
        out[:, start:end] = multi_head_attention_vectorized(
            qh[:, start:end], kh[:, start:end], vh[:, start:end], mask=mask
        )

    return out
