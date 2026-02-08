import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import depthwise_separable_chunked


def test_chunked_large_batch_runs():
    x = np.random.randn(12, 8, 32, 32).astype(np.float32)
    dw = np.random.randn(8, 3, 3).astype(np.float32)
    pw = np.random.randn(16, 8).astype(np.float32)
    y = depthwise_separable_chunked(x, dw, pw, stride=1, padding=1, chunk_size=3)
    assert y.shape == (12, 16, 32, 32)
