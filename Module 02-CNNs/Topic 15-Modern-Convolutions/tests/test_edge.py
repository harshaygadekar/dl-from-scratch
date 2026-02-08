import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import depthwise_conv2d


def test_depthwise_channel_mismatch_raises():
    x = np.random.randn(1, 3, 8, 8).astype(np.float32)
    w = np.random.randn(4, 3, 3).astype(np.float32)
    with pytest.raises(ValueError):
        depthwise_conv2d(x, w)
