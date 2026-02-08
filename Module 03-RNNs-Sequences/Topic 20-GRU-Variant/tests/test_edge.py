import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import gru_step


def test_zero_weights_stable_output():
    b, d, h = 2, 3, 4
    x = np.zeros((b, d), dtype=np.float32)
    h_prev = np.zeros((b, h), dtype=np.float32)
    w_x = np.zeros((d, 3 * h), dtype=np.float32)
    w_h = np.zeros((h, 3 * h), dtype=np.float32)
    b_vec = np.zeros(3 * h, dtype=np.float32)
    out = gru_step(x, h_prev, w_x, w_h, b_vec)
    assert np.isfinite(out).all()
