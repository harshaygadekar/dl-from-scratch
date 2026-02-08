import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import sequence_mask, bidirectional_rnn_forward


def test_sequence_mask_values():
    lengths = np.array([1, 3, 2])
    mask = sequence_mask(lengths, max_len=4)
    expected = np.array([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(mask, expected)


def test_bidirectional_output_shape():
    t, b, d, h = 5, 2, 4, 3
    x = np.random.randn(t, b, d).astype(np.float32)
    h0f = np.zeros((b, h), dtype=np.float32)
    h0b = np.zeros((b, h), dtype=np.float32)
    wxf = np.random.randn(d, h).astype(np.float32)
    whf = np.random.randn(h, h).astype(np.float32)
    bf = np.zeros(h, dtype=np.float32)
    wxb = np.random.randn(d, h).astype(np.float32)
    whb = np.random.randn(h, h).astype(np.float32)
    bb = np.zeros(h, dtype=np.float32)
    out = bidirectional_rnn_forward(x, h0f, h0b, wxf, whf, bf, wxb, whb, bb)
    assert out.shape == (t, b, 2 * h)
