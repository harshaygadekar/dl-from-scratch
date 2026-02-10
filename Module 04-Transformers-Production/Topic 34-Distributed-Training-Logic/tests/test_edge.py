import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import sync_sgd_step


def test_sync_sgd_step_dict_keys_preserved():
    params = {"w": np.ones((2, 2), dtype=np.float32), "b": np.zeros((2,), dtype=np.float32)}
    grads = [
        {"w": np.ones((2, 2), dtype=np.float32), "b": np.ones((2,), dtype=np.float32)},
        {"w": np.zeros((2, 2), dtype=np.float32), "b": np.zeros((2,), dtype=np.float32)},
    ]
    out = sync_sgd_step(params, grads, lr=0.1)
    assert set(out.keys()) == {"w", "b"}



def test_phase_c_edge_34_sync_sgd_expected_numeric():
    params = {"w": np.array([2.0, -2.0], dtype=np.float32)}
    grads = [
        {"w": np.array([1.0, 3.0], dtype=np.float32)},
        {"w": np.array([3.0, 1.0], dtype=np.float32)},
    ]
    out = sync_sgd_step(params, grads, lr=0.5)
    np.testing.assert_allclose(out["w"], np.array([1.0, -3.0], dtype=np.float32))

