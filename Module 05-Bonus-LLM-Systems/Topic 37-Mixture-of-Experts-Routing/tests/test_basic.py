import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level01_naive import top1_route, top2_route


def test_top1_shape():
    x = np.random.randn(32, 4).astype(np.float32)
    y = top1_route(x)
    assert y.shape == (32,)


def test_top2_shape():
    x = np.random.randn(32, 6).astype(np.float32)
    y = top2_route(x)
    assert y.shape == (32, 2)
