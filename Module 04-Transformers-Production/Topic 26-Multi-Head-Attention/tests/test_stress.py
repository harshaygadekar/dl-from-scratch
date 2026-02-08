import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import multi_head_attention_vectorized


def test_vectorized_mha_runs():
    qh = np.random.randn(4, 8, 64, 16).astype(np.float32)
    kh = np.random.randn(4, 8, 64, 16).astype(np.float32)
    vh = np.random.randn(4, 8, 64, 16).astype(np.float32)
    out = multi_head_attention_vectorized(qh, kh, vh)
    assert out.shape == (4, 8, 64, 16)
