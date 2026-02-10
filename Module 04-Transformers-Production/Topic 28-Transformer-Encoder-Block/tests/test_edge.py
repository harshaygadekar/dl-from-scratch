import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import layer_norm


def test_layer_norm_finite_for_constant_input():
    x = np.ones((2, 3, 4), dtype=np.float32)
    y = layer_norm(x)
    assert np.isfinite(y).all()



def test_phase_c_edge_28_layer_norm_large_values_finite():
    x = np.full((2, 3, 4), 1e6, dtype=np.float32)
    y = layer_norm(x)
    assert np.isfinite(y).all()

