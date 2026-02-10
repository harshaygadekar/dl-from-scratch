import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import chunked_attention_scores


def test_chunked_scores_runs():
    q = np.random.randn(4, 128, 64).astype(np.float32)
    k = np.random.randn(4, 128, 64).astype(np.float32)
    s = chunked_attention_scores(q, k, chunk_size=32)
    assert s.shape == (4, 128, 128)



def test_phase_c_regression_25_chunked_matches_full_matmul():
    np.random.seed(7)
    q = np.random.randn(2, 32, 16).astype(np.float32)
    k = np.random.randn(2, 32, 16).astype(np.float32)
    full = np.matmul(q, np.swapaxes(k, -1, -2))
    chunked = chunked_attention_scores(q, k, chunk_size=8)
    np.testing.assert_allclose(chunked, full, rtol=1e-5, atol=1e-5)

