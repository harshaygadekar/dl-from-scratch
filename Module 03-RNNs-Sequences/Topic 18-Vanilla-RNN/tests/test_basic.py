import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import rnn_step, rnn_forward
from level02_vectorized import rnn_forward_batch_first


def test_rnn_step_shape():
    x = np.random.randn(4, 6).astype(np.float32)
    h = np.random.randn(4, 5).astype(np.float32)
    w_xh = np.random.randn(6, 5).astype(np.float32)
    w_hh = np.random.randn(5, 5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    out = rnn_step(x, h, w_xh, w_hh, b)
    assert out.shape == (4, 5)


def test_rnn_forward_shape():
    x = np.random.randn(7, 3, 6).astype(np.float32)
    h0 = np.zeros((3, 5), dtype=np.float32)
    w_xh = np.random.randn(6, 5).astype(np.float32)
    w_hh = np.random.randn(5, 5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    out = rnn_forward(x, h0, w_xh, w_hh, b)
    assert out.shape == (7, 3, 5)


def test_batch_first_wrapper_shape():
    x = np.random.randn(3, 7, 6).astype(np.float32)
    h0 = np.zeros((3, 5), dtype=np.float32)
    w_xh = np.random.randn(6, 5).astype(np.float32)
    w_hh = np.random.randn(5, 5).astype(np.float32)
    b = np.random.randn(5).astype(np.float32)
    out = rnn_forward_batch_first(x, h0, w_xh, w_hh, b)
    assert out.shape == (3, 7, 5)
