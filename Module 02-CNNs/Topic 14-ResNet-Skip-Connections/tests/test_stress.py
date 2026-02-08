import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import residual_stack


def test_two_block_stack_runs():
    x = np.random.randn(4, 16, 16, 16).astype(np.float32)
    blocks = [
        {
            "w1": np.random.randn(16, 16).astype(np.float32),
            "w2": np.random.randn(16, 16).astype(np.float32),
        },
        {
            "w1": np.random.randn(16, 16).astype(np.float32),
            "w2": np.random.randn(16, 16).astype(np.float32),
        },
    ]
    y = residual_stack(x, blocks)
    assert y.shape == x.shape
