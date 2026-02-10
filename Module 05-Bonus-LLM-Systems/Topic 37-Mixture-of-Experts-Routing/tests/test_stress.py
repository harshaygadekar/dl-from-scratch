import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level03_memory_efficient import load_balance_penalty


def test_penalty_finite_large_counts():
    counts = np.random.randint(0, 1000, size=(64,))
    p = load_balance_penalty(counts)
    assert np.isfinite(p)
