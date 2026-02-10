import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level02_vectorized import selective_scan_vectorized


def test_zero_input_stays_zero():
    a = np.ones(16, dtype=np.float32) * 0.5
    b = np.ones(16, dtype=np.float32) * 0.5
    x = np.zeros(16, dtype=np.float32)
    y = selective_scan_vectorized(a, b, x)
    np.testing.assert_allclose(y, 0.0)
