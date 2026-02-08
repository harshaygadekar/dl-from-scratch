import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import max_pool2d_forward, avg_pool2d_forward
from level02_vectorized import max_pool2d_vectorized, avg_pool2d_vectorized


def test_max_pool_known_values():
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    out, _ = max_pool2d_forward(x, kernel_size=2, stride=2)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == 4


def test_avg_pool_known_values():
    x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
    out = avg_pool2d_forward(x, kernel_size=2, stride=2)
    assert np.isclose(out[0, 0, 0, 0], 2.5)


def test_vectorized_matches_naive():
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    y1, _ = max_pool2d_forward(x, kernel_size=2, stride=2)
    y2 = max_pool2d_vectorized(x, kernel_size=2, stride=2)
    np.testing.assert_allclose(y1, y2)

    a1 = avg_pool2d_forward(x, kernel_size=2, stride=2)
    a2 = avg_pool2d_vectorized(x, kernel_size=2, stride=2)
    np.testing.assert_allclose(a1, a2)
