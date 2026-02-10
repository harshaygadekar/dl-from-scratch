import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level01_naive import selective_scan_naive
from level03_memory_efficient import chunked_selective_scan


def test_chunked_matches_naive():
    np.random.seed(7)
    n = 4096
    a = np.random.uniform(0.8, 0.99, size=n).astype(np.float32)
    b = np.random.uniform(0.01, 0.2, size=n).astype(np.float32)
    x = np.random.randn(n).astype(np.float32)
    y1 = selective_scan_naive(a, b, x)
    y2 = chunked_selective_scan(a, b, x, chunk=256)
    np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)
