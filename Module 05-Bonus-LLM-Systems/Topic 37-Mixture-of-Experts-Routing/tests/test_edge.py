import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level02_vectorized import route_with_capacity


def test_capacity_respected():
    top1 = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    _, counts = route_with_capacity(top1, num_experts=3, capacity=2)
    assert np.all(counts <= 2)
