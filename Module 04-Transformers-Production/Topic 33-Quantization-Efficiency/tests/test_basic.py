import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import symmetric_quant_params, quantize_int8, dequantize_int8, fake_quantize


def test_quant_dequant_shape_preserved():
    x = np.random.randn(16, 16).astype(np.float32)
    s = symmetric_quant_params(x)
    q = quantize_int8(x, s)
    y = dequantize_int8(q, s)
    assert q.shape == x.shape
    assert y.shape == x.shape


def test_fake_quant_finite():
    x = np.random.randn(32).astype(np.float32)
    s = symmetric_quant_params(x)
    y = fake_quantize(x, s)
    assert np.isfinite(y).all()



def test_phase_c_basic_33_roundtrip_error_bound():
    np.random.seed(41)
    x = np.random.randn(128).astype(np.float32)
    s = symmetric_quant_params(x)
    y = dequantize_int8(quantize_int8(x, s), s)
    mae = np.mean(np.abs(x - y))
    assert mae < 0.03

