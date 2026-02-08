import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import gru_forward


def test_gru_medium_sequence_runs():
    t, b, d, h = 64, 12, 16, 32
    x = np.random.randn(t, b, d).astype(np.float32)
    h0 = np.zeros((b, h), dtype=np.float32)
    w_x = np.random.randn(d, 3 * h).astype(np.float32) * 0.1
    w_h = np.random.randn(h, 3 * h).astype(np.float32) * 0.1
    bias = np.zeros(3 * h, dtype=np.float32)
    out = gru_forward(x, h0, w_x, w_h, bias)
    assert out.shape == (t, b, h)
