import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import lstm_step


def test_zero_inputs_do_not_nan():
    b, d, h = 2, 3, 4
    x = np.zeros((b, d), dtype=np.float32)
    h0 = np.zeros((b, h), dtype=np.float32)
    c0 = np.zeros((b, h), dtype=np.float32)
    w_x = np.zeros((d, 4 * h), dtype=np.float32)
    w_h = np.zeros((h, 4 * h), dtype=np.float32)
    bias = np.zeros(4 * h, dtype=np.float32)
    h1, c1 = lstm_step(x, h0, c0, w_x, w_h, bias)
    assert np.isfinite(h1).all()
    assert np.isfinite(c1).all()
