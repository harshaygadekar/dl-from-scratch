import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import group_norm_nchw


def test_group_count_must_divide_channels():
    x = np.random.randn(2, 10, 4, 4).astype(np.float32)
    with pytest.raises(ValueError):
        group_norm_nchw(x, num_groups=4)
