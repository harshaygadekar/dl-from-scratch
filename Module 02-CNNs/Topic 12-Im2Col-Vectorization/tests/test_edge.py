import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import conv2d_im2col_naive


def test_1x1_kernel_keeps_spatial_size():
    x = np.random.randn(1, 5, 7, 9).astype(np.float32)
    w = np.random.randn(3, 5, 1, 1).astype(np.float32)
    y = conv2d_im2col_naive(x, w, stride=1, padding=0)
    assert y.shape == (1, 3, 7, 9)


def test_stride_two_reduces_resolution():
    x = np.random.randn(1, 1, 8, 8).astype(np.float32)
    w = np.random.randn(1, 1, 3, 3).astype(np.float32)
    y = conv2d_im2col_naive(x, w, stride=2, padding=1)
    assert y.shape == (1, 1, 4, 4)
