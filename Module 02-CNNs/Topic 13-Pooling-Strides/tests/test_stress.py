import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import max_pool2d_vectorized


def test_large_pool_runs():
    x = np.random.randn(8, 64, 64, 64).astype(np.float32)
    y = max_pool2d_vectorized(x, kernel_size=2, stride=2)
    assert y.shape == (8, 64, 32, 32)
