import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import bahdanau_context


def test_attention_shapes_and_normalization():
    t, b, h_enc, h_dec, a = 7, 3, 5, 6, 4
    query = np.random.randn(b, h_dec).astype(np.float32)
    keys = np.random.randn(t, b, h_enc).astype(np.float32)
    values = np.random.randn(t, b, h_enc).astype(np.float32)

    w_q = np.random.randn(h_dec, a).astype(np.float32)
    w_k = np.random.randn(h_enc, a).astype(np.float32)
    v_a = np.random.randn(a).astype(np.float32)
    b_a = np.zeros(a, dtype=np.float32)

    context, weights = bahdanau_context(query, keys, values, w_q, w_k, v_a, b_a)
    assert context.shape == (b, h_enc)
    assert weights.shape == (b, t)
    np.testing.assert_allclose(weights.sum(axis=1), np.ones(b), atol=1e-5)
