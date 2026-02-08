import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import im2col_naive, conv2d_im2col_naive
from level02_vectorized import im2col_vectorized, conv2d_im2col_vectorized


def test_im2col_shape_match():
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    cols_naive, h1, w1 = im2col_naive(x, kernel_size=3, stride=1, padding=1)
    cols_vect, h2, w2 = im2col_vectorized(x, kernel_size=3, stride=1, padding=1)
    assert cols_naive.shape == cols_vect.shape
    assert (h1, w1) == (h2, w2)


def test_conv_outputs_close():
    np.random.seed(0)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    w = np.random.randn(4, 3, 3, 3).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)

    y1 = conv2d_im2col_naive(x, w, b, stride=1, padding=1)
    y2 = conv2d_im2col_vectorized(x, w, b, stride=1, padding=1)
    np.testing.assert_allclose(y1, y2, rtol=1e-5, atol=1e-5)
