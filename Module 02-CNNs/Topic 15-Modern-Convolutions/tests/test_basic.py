import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import depthwise_conv2d, pointwise_conv2d, dilated_conv2d
from level02_vectorized import depthwise_separable_conv2d


def test_depthwise_shape():
    x = np.random.randn(2, 4, 10, 10).astype(np.float32)
    w = np.random.randn(4, 3, 3).astype(np.float32)
    y = depthwise_conv2d(x, w, stride=1, padding=1)
    assert y.shape == (2, 4, 10, 10)


def test_pointwise_channel_projection():
    x = np.random.randn(1, 4, 8, 8).astype(np.float32)
    w = np.random.randn(6, 4).astype(np.float32)
    y = pointwise_conv2d(x, w)
    assert y.shape == (1, 6, 8, 8)


def test_depthwise_separable_shape():
    x = np.random.randn(1, 4, 8, 8).astype(np.float32)
    dw = np.random.randn(4, 3, 3).astype(np.float32)
    pw = np.random.randn(7, 4).astype(np.float32)
    y = depthwise_separable_conv2d(x, dw, pw, stride=1, padding=1)
    assert y.shape == (1, 7, 8, 8)


def test_dilated_shape():
    x = np.random.randn(1, 3, 12, 12).astype(np.float32)
    w = np.random.randn(5, 3, 3, 3).astype(np.float32)
    y = dilated_conv2d(x, w, dilation=2, stride=1, padding=2)
    assert y.shape == (1, 5, 12, 12)
