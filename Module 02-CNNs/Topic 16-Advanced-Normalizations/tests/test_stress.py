import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import group_norm_nchw


def test_large_group_norm_runs():
    x = np.random.randn(8, 32, 32, 32).astype(np.float32)
    y = group_norm_nchw(x, num_groups=8)
    assert y.shape == x.shape
