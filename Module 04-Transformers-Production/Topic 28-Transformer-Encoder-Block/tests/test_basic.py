import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import encoder_block_forward


def test_encoder_block_shape():
    b, t, d, h = 2, 6, 16, 32
    x = np.random.randn(b, t, d).astype(np.float32)
    block = {
        "w_q": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_k": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_v": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_o": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w1": np.random.randn(d, h).astype(np.float32) * 0.1,
        "b1": np.zeros(h, dtype=np.float32),
        "w2": np.random.randn(h, d).astype(np.float32) * 0.1,
        "b2": np.zeros(d, dtype=np.float32),
    }
    y = encoder_block_forward(x, **block)
    assert y.shape == x.shape
