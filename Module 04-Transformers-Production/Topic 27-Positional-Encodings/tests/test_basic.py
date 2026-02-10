import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import sinusoidal_positional_encoding, add_positional_encoding


def test_sinusoidal_shape():
    pe = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    assert pe.shape == (10, 16)


def test_add_pe_shape():
    x = np.random.randn(2, 10, 16).astype(np.float32)
    pe = sinusoidal_positional_encoding(10, 16)
    y = add_positional_encoding(x, pe)
    assert y.shape == x.shape



def test_phase_c_numeric_27_first_position_values():
    pe = sinusoidal_positional_encoding(seq_len=4, d_model=8)
    np.testing.assert_allclose(pe[0, 0::2], np.zeros(4), atol=1e-7)
    np.testing.assert_allclose(pe[0, 1::2], np.ones(4), atol=1e-7)

