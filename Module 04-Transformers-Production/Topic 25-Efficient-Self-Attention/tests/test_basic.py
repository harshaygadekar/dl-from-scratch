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



def test_phase_c_determinism_25_attention_values():
    np.random.seed(42)
    q = np.random.randn(2, 4, 8).astype(np.float32)
    k = np.random.randn(2, 4, 8).astype(np.float32)
    v = np.random.randn(2, 4, 8).astype(np.float32)
    out1, w1 = scaled_dot_product_attention(q, k, v)
    out2, w2 = scaled_dot_product_attention(q, k, v)
    np.testing.assert_allclose(out1, out2, atol=1e-6)
    np.testing.assert_allclose(w1, w2, atol=1e-6)

