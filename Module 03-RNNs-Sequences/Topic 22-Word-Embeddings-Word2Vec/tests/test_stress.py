import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import negative_sampling_loss_batch


def test_batch_loss_runs():
    b, d, k = 64, 32, 10
    center = np.random.randn(b, d).astype(np.float32)
    pos = np.random.randn(b, d).astype(np.float32)
    neg = np.random.randn(b, k, d).astype(np.float32)
    loss = negative_sampling_loss_batch(center, pos, neg)
    assert np.isfinite(loss)
