import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level02_vectorized import quantize_4bit_linear, dequantize_4bit_linear


def test_quantize_clip_range():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
    q = quantize_4bit_linear(x, scale=0.5)
    assert q.min() >= -8
    assert q.max() <= 7


def test_dequantize_finite():
    q = np.array([-8, -1, 0, 7], dtype=np.int8)
    y = dequantize_4bit_linear(q, scale=0.25)
    assert np.isfinite(y).all()
