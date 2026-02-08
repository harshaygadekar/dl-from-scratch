import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import rnn_forward


def test_long_sequence_runs():
    t, b, d, h = 64, 16, 32, 64
    x = np.random.randn(t, b, d).astype(np.float32)
    h0 = np.zeros((b, h), dtype=np.float32)
    w_xh = np.random.randn(d, h).astype(np.float32)
    w_hh = np.random.randn(h, h).astype(np.float32)
    bias = np.zeros(h, dtype=np.float32)
    out = rnn_forward(x, h0, w_xh, w_hh, bias)
    assert out.shape == (t, b, h)
