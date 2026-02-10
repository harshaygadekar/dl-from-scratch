import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import multi_head_attention_vectorized


def test_vectorized_mha_runs():
    qh = np.random.randn(4, 8, 64, 16).astype(np.float32)
    kh = np.random.randn(4, 8, 64, 16).astype(np.float32)
    vh = np.random.randn(4, 8, 64, 16).astype(np.float32)
    out = multi_head_attention_vectorized(qh, kh, vh)
    assert out.shape == (4, 8, 64, 16)


from level03_memory_efficient import mha_head_chunks


def test_phase_c_regression_26_chunked_heads_match_vectorized():
    np.random.seed(11)
    qh = np.random.randn(2, 6, 12, 8).astype(np.float32)
    kh = np.random.randn(2, 6, 12, 8).astype(np.float32)
    vh = np.random.randn(2, 6, 12, 8).astype(np.float32)
    full = multi_head_attention_vectorized(qh, kh, vh)
    chunked = mha_head_chunks(qh, kh, vh, head_chunk=2)
    np.testing.assert_allclose(chunked, full, rtol=1e-5, atol=1e-5)

