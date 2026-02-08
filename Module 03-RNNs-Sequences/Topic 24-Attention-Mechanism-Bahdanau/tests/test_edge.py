import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import bahdanau_context


def test_mask_excludes_padded_positions():
    t, b, h_enc, h_dec, a = 5, 2, 4, 4, 3
    query = np.random.randn(b, h_dec).astype(np.float32)
    keys = np.random.randn(t, b, h_enc).astype(np.float32)
    values = np.random.randn(t, b, h_enc).astype(np.float32)

    w_q = np.random.randn(h_dec, a).astype(np.float32)
    w_k = np.random.randn(h_enc, a).astype(np.float32)
    v_a = np.random.randn(a).astype(np.float32)
    b_a = np.zeros(a, dtype=np.float32)

    mask = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=np.float32)
    _, weights = bahdanau_context(query, keys, values, w_q, w_k, v_a, b_a, mask=mask)

    assert np.all(weights[0, 2:] < 1e-6)
    assert np.all(weights[1, 3:] < 1e-6)
