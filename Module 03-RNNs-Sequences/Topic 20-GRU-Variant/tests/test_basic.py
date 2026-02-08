import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import gru_step, gru_forward


def test_gru_step_shape():
    b, d, h = 4, 6, 5
    x = np.random.randn(b, d).astype(np.float32)
    h_prev = np.random.randn(b, h).astype(np.float32)
    w_x = np.random.randn(d, 3 * h).astype(np.float32)
    w_h = np.random.randn(h, 3 * h).astype(np.float32)
    b_vec = np.random.randn(3 * h).astype(np.float32)
    h_t = gru_step(x, h_prev, w_x, w_h, b_vec)
    assert h_t.shape == (b, h)


def test_gru_forward_shape():
    t, b, d, h = 9, 3, 4, 6
    x = np.random.randn(t, b, d).astype(np.float32)
    h0 = np.zeros((b, h), dtype=np.float32)
    w_x = np.random.randn(d, 3 * h).astype(np.float32)
    w_h = np.random.randn(h, 3 * h).astype(np.float32)
    bias = np.zeros(3 * h, dtype=np.float32)
    out = gru_forward(x, h0, w_x, w_h, bias)
    assert out.shape == (t, b, h)
