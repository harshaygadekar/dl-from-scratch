import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import make_batches


def test_make_batches_raises_for_short_sequence():
    tokens = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        make_batches(tokens, block_size=16, batch_size=2)



def test_phase_c_edge_30_batch_targets_are_shifted():
    np.random.seed(29)
    tokens = np.arange(200, dtype=np.int64)
    x, y = make_batches(tokens, block_size=12, batch_size=3)
    np.testing.assert_array_equal(y, x + 1)

