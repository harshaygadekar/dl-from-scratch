import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import split_heads, combine_heads, multi_head_attention


def test_split_combine_roundtrip():
    x = np.random.randn(2, 5, 16).astype(np.float32)
    heads = split_heads(x, num_heads=4)
    y = combine_heads(heads)
    np.testing.assert_allclose(x, y, atol=1e-6)


def test_mha_output_shape():
    b, t, d, h = 2, 6, 16, 4
    x = np.random.randn(b, t, d).astype(np.float32)
    w = np.random.randn(d, d).astype(np.float32) * 0.1
    out = multi_head_attention(x, w, w, w, w, num_heads=h)
    assert out.shape == (b, t, d)



def test_phase_c_shape_26_split_expected_layout():
    x = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
    heads = split_heads(x, num_heads=2)
    assert heads.shape == (2, 2, 3, 4)
    back = combine_heads(heads)
    np.testing.assert_allclose(back, x)

