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
