import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import apply_mask_batch_first


def test_apply_mask_zeroes_padded_positions():
    x = np.ones((2, 4, 3), dtype=np.float32)
    mask = np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    y = apply_mask_batch_first(x, mask)
    assert np.all(y[0, 2:] == 0)
    assert np.all(y[1, 1:] == 0)
