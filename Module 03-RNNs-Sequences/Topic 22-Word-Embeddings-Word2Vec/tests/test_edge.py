import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import subsample_tokens


def test_subsample_returns_subset_or_equal():
    tokens = [1, 2, 3, 4, 5] * 5
    freqs = {1: 0.2, 2: 0.1, 3: 0.05, 4: 0.01, 5: 0.005}
    out = subsample_tokens(tokens, freqs, threshold=1e-3)
    assert len(out) <= len(tokens)
