import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import scaled_dot_product_attention


def test_attention_shapes_and_normalization():
    b, t, d = 2, 5, 8
    q = np.random.randn(b, t, d).astype(np.float32)
    k = np.random.randn(b, t, d).astype(np.float32)
    v = np.random.randn(b, t, d).astype(np.float32)

    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (b, t, d)
    assert w.shape == (b, t, t)
    np.testing.assert_allclose(w.sum(axis=-1), np.ones((b, t)), atol=1e-5)
