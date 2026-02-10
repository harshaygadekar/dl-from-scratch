import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import chunked_ffn


def test_chunked_ffn_runs():
    b, t, d, h = 4, 256, 64, 128
    x = np.random.randn(b, t, d).astype(np.float32)
    w1 = np.random.randn(d, h).astype(np.float32) * 0.1
    b1 = np.zeros(h, dtype=np.float32)
    w2 = np.random.randn(h, d).astype(np.float32) * 0.1
    b2 = np.zeros(d, dtype=np.float32)
    y = chunked_ffn(x, w1, b1, w2, b2, chunk_size=64)
    assert y.shape == x.shape



def test_phase_c_regression_28_chunked_ffn_matches_dense():
    np.random.seed(5)
    b, t, d, h = 2, 23, 12, 20
    x = np.random.randn(b, t, d).astype(np.float32)
    w1 = np.random.randn(d, h).astype(np.float32) * 0.1
    b1 = np.random.randn(h).astype(np.float32) * 0.01
    w2 = np.random.randn(h, d).astype(np.float32) * 0.1
    b2 = np.random.randn(d).astype(np.float32) * 0.01
    dense = np.maximum(0, x @ w1 + b1) @ w2 + b2
    chunked = chunked_ffn(x, w1, b1, w2, b2, chunk_size=7)
    np.testing.assert_allclose(chunked, dense, rtol=1e-5, atol=1e-5)

