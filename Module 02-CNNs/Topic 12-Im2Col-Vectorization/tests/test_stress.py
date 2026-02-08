import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import conv2d_im2col_vectorized


def test_large_tensor_runs():
    x = np.random.randn(4, 16, 32, 32).astype(np.float32)
    w = np.random.randn(32, 16, 3, 3).astype(np.float32)
    y = conv2d_im2col_vectorized(x, w, stride=1, padding=1)
    assert y.shape == (4, 32, 32, 32)
