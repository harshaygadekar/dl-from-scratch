import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import apply_rope


def test_rope_requires_even_dimension():
    x = np.random.randn(1, 4, 15).astype(np.float32)
    with pytest.raises(ValueError):
        apply_rope(x)
