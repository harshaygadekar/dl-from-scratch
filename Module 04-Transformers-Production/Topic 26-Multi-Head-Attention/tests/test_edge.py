import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import split_heads


def test_split_heads_invalid_dimension_raises():
    x = np.random.randn(1, 3, 10).astype(np.float32)
    with pytest.raises(ValueError):
        split_heads(x, num_heads=3)
