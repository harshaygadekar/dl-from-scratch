import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import symmetric_quant_params


def test_zero_tensor_has_valid_scale():
    x = np.zeros((8, 8), dtype=np.float32)
    s = symmetric_quant_params(x)
    assert s > 0


from level02_vectorized import per_channel_scales


def test_phase_c_edge_33_per_channel_scale_shape():
    w = np.random.randn(8, 16).astype(np.float32)
    s = per_channel_scales(w, axis=1)
    assert s.shape == (8, 1)
    assert np.all(s > 0)

