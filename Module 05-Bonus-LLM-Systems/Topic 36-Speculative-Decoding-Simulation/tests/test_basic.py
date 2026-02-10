import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level01_naive import greedy_next, propose_draft


def test_greedy_next_returns_argmax():
    logits = np.array([0.1, 0.9, -0.2], dtype=np.float32)
    assert greedy_next(logits) == 1


def test_propose_draft_length():
    logits = np.array([0.1, 0.9, 0.4, 0.2], dtype=np.float32)
    toks = propose_draft(logits, k=2)
    assert len(toks) == 2
