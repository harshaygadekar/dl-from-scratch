import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import trim_to_max_length


def test_trim_reduces_time_dimension():
    x = np.random.randn(16, 128, 32).astype(np.float32)
    lengths = np.random.randint(32, 128, size=(16,))
    y = trim_to_max_length(x, lengths)
    assert y.shape[1] == int(lengths.max())
