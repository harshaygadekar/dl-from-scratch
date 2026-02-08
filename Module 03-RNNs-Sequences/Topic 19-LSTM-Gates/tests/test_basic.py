import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import lstm_step, lstm_forward


def test_lstm_step_shapes():
    b, d, h = 4, 6, 5
    x = np.random.randn(b, d).astype(np.float32)
    h0 = np.zeros((b, h), dtype=np.float32)
    c0 = np.zeros((b, h), dtype=np.float32)
    w_x = np.random.randn(d, 4 * h).astype(np.float32)
    w_h = np.random.randn(h, 4 * h).astype(np.float32)
    bias = np.random.randn(4 * h).astype(np.float32)

    h1, c1 = lstm_step(x, h0, c0, w_x, w_h, bias)
    assert h1.shape == (b, h)
    assert c1.shape == (b, h)


def test_lstm_forward_shapes():
    t, b, d, h = 7, 3, 6, 5
    x = np.random.randn(t, b, d).astype(np.float32)
    h0 = np.zeros((b, h), dtype=np.float32)
    c0 = np.zeros((b, h), dtype=np.float32)
    w_x = np.random.randn(d, 4 * h).astype(np.float32)
    w_h = np.random.randn(h, 4 * h).astype(np.float32)
    bias = np.zeros(4 * h, dtype=np.float32)

    h_out, c_out = lstm_forward(x, h0, c0, w_x, w_h, bias)
    assert h_out.shape == (t, b, h)
    assert c_out.shape == (t, b, h)
