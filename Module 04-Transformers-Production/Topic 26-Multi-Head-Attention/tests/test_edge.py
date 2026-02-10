import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import split_heads


def test_split_heads_invalid_dimension_raises():
    x = np.random.randn(1, 3, 10).astype(np.float32)
    with pytest.raises(ValueError):
        split_heads(x, num_heads=3)


from level02_vectorized import multi_head_attention_vectorized


def test_phase_c_edge_26_mask_rank_equivalence():
    np.random.seed(1)
    qh = np.random.randn(1, 2, 4, 8).astype(np.float32)
    kh = np.random.randn(1, 2, 4, 8).astype(np.float32)
    vh = np.random.randn(1, 2, 4, 8).astype(np.float32)
    mask2 = np.tril(np.ones((4, 4), dtype=np.float32))
    mask3 = np.repeat(mask2[None, :, :], 1, axis=0)
    y2 = multi_head_attention_vectorized(qh, kh, vh, mask=mask2)
    y3 = multi_head_attention_vectorized(qh, kh, vh, mask=mask3)
    np.testing.assert_allclose(y2, y3, atol=1e-6)

