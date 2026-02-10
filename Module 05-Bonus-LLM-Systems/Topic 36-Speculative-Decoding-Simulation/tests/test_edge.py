import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level02_vectorized import speculative_accept_reject


def test_accept_reject_prefix_property():
    logits = np.array([2.0, 1.0, 0.5, -0.5], dtype=np.float32)
    draft = np.array([0, 1, 2, 3], dtype=np.int64)
    accepted = speculative_accept_reject(logits, draft)
    assert len(accepted) <= len(draft)
    assert accepted == list(draft[: len(accepted)])
