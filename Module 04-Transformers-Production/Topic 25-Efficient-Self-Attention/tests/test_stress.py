import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import chunked_attention_scores


def test_chunked_scores_runs():
    q = np.random.randn(4, 128, 64).astype(np.float32)
    k = np.random.randn(4, 128, 64).astype(np.float32)
    s = chunked_attention_scores(q, k, chunk_size=32)
    assert s.shape == (4, 128, 128)
