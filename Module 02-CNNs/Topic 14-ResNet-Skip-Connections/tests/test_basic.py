import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import conv1x1, residual_block_forward


def test_conv1x1_shape():
    x = np.random.randn(2, 8, 16, 16).astype(np.float32)
    w = np.random.randn(12, 8).astype(np.float32)
    y = conv1x1(x, w)
    assert y.shape == (2, 12, 16, 16)


def test_identity_residual_shape_preserved():
    x = np.random.randn(2, 8, 8, 8).astype(np.float32)
    w1 = np.random.randn(8, 8).astype(np.float32)
    w2 = np.random.randn(8, 8).astype(np.float32)
    y = residual_block_forward(x, w1=w1, w2=w2)
    assert y.shape == x.shape
