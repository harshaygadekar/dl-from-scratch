import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import symmetric_quant_params


def test_zero_tensor_has_valid_scale():
    x = np.zeros((8, 8), dtype=np.float32)
    s = symmetric_quant_params(x)
    assert s > 0
