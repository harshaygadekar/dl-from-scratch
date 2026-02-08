import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import max_pool2d_forward, max_pool2d_backward


def test_backward_shape_matches_input():
    x = np.random.randn(1, 2, 6, 6).astype(np.float32)
    out, cache = max_pool2d_forward(x, kernel_size=2, stride=2)
    grad_out = np.ones_like(out)
    grad_in = max_pool2d_backward(grad_out, cache)
    assert grad_in.shape == x.shape


def test_non_square_input():
    x = np.random.randn(1, 1, 6, 10).astype(np.float32)
    out, _ = max_pool2d_forward(x, kernel_size=2, stride=2)
    assert out.shape == (1, 1, 3, 5)
