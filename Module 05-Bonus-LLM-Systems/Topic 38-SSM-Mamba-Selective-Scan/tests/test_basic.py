import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level01_naive import selective_scan_naive


def test_scan_shape_preserved():
    a = np.ones(32, dtype=np.float32) * 0.9
    b = np.ones(32, dtype=np.float32) * 0.1
    x = np.random.randn(32).astype(np.float32)
    y = selective_scan_naive(a, b, x)
    assert y.shape == x.shape
