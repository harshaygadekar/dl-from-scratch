import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import truncate_prefix


def test_truncate_prefix_window():
    x = np.random.randn(2, 20, 8).astype(np.float32)
    y = truncate_prefix(x, max_context=12)
    assert y.shape[1] == 12



def test_phase_c_edge_29_truncate_prefix_noop_when_short():
    x = np.random.randn(2, 5, 8).astype(np.float32)
    y = truncate_prefix(x, max_context=10)
    np.testing.assert_allclose(y, x)

