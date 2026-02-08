import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import residual_block_forward


def test_projection_shortcut_channel_change():
    x = np.random.randn(1, 8, 6, 6).astype(np.float32)
    w1 = np.random.randn(16, 8).astype(np.float32)
    w2 = np.random.randn(16, 16).astype(np.float32)
    proj_w = np.random.randn(16, 8).astype(np.float32)
    y = residual_block_forward(x, w1=w1, w2=w2, proj_w=proj_w)
    assert y.shape == (1, 16, 6, 6)
