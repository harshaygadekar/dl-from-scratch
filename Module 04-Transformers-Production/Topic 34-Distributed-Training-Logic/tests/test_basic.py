import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import split_batch, average_gradients, sgd_update


def test_split_batch_count():
    x = np.random.randn(17, 8).astype(np.float32)
    chunks = split_batch(x, num_workers=4)
    assert len(chunks) == 4
    assert sum(c.shape[0] for c in chunks) == 17


def test_average_gradients():
    g1 = np.ones((3, 3), dtype=np.float32)
    g2 = np.zeros((3, 3), dtype=np.float32)
    g = average_gradients([g1, g2])
    np.testing.assert_allclose(g, 0.5)


def test_sgd_update_changes_param():
    p = np.array([1.0, 2.0], dtype=np.float32)
    g = np.array([0.5, 0.5], dtype=np.float32)
    p2 = sgd_update(p, g, lr=0.1)
    assert np.all(p2 < p)



def test_phase_c_basic_34_average_gradients_three_workers():
    g = average_gradients([
        np.full((2, 2), 1.0, dtype=np.float32),
        np.full((2, 2), 3.0, dtype=np.float32),
        np.full((2, 2), 5.0, dtype=np.float32),
    ])
    np.testing.assert_allclose(g, 3.0)

