import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import decode_greedy


def test_greedy_decode_runs():
    b, h, v = 16, 32, 50
    h_init = np.zeros((b, h), dtype=np.float32)
    w_xh = np.random.randn(v, h).astype(np.float32) * 0.01
    w_hh = np.random.randn(h, h).astype(np.float32) * 0.01
    b_h = np.zeros(h, dtype=np.float32)
    w_out = np.random.randn(h, v).astype(np.float32) * 0.01
    b_out = np.zeros(v, dtype=np.float32)
    out = decode_greedy(0, h_init, steps=20, vocab_size=v, w_xh=w_xh, w_hh=w_hh, b_h=b_h, w_out=w_out, b_out=b_out)
    assert out.shape == (20, b)
