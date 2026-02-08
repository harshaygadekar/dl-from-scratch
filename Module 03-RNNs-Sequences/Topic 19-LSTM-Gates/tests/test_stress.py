import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import lstm_forward


def test_lstm_long_sequence_runs():
    t, b, d, h = 48, 8, 32, 64
    x = np.random.randn(t, b, d).astype(np.float32)
    h0 = np.zeros((b, h), dtype=np.float32)
    c0 = np.zeros((b, h), dtype=np.float32)
    w_x = np.random.randn(d, 4 * h).astype(np.float32) * 0.1
    w_h = np.random.randn(h, 4 * h).astype(np.float32) * 0.1
    bias = np.zeros(4 * h, dtype=np.float32)
    h_out, c_out = lstm_forward(x, h0, c0, w_x, w_h, bias)
    assert h_out.shape == (t, b, h)
    assert c_out.shape == (t, b, h)
