import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import blockwise_qk_scores


def test_blockwise_scores_runs():
    q = np.random.randn(2, 256, 64).astype(np.float32)
    k = np.random.randn(2, 256, 64).astype(np.float32)
    s = blockwise_qk_scores(q, k, block=64)
    assert s.shape == (2, 256, 256)



def test_phase_c_stress_32_blockwise_matches_full_scores():
    np.random.seed(37)
    q = np.random.randn(2, 96, 32).astype(np.float32)
    k = np.random.randn(2, 96, 32).astype(np.float32)
    full = np.matmul(q, np.swapaxes(k, -1, -2))
    blk = blockwise_qk_scores(q, k, block=24)
    np.testing.assert_allclose(blk, full, rtol=1e-5, atol=1e-5)

