import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import init_kv_cache, append_kv, current_kv


def test_cache_append_increments_length():
    cache = init_kv_cache(batch=2, heads=4, max_seq=16, head_dim=8)
    k = np.random.randn(2, 4, 1, 8).astype(np.float32)
    v = np.random.randn(2, 4, 1, 8).astype(np.float32)
    cache = append_kv(cache, k, v)
    assert cache["len"] == 1
    k_now, v_now = current_kv(cache)
    assert k_now.shape == (2, 4, 1, 8)
    assert v_now.shape == (2, 4, 1, 8)



def test_phase_c_basic_31_append_preserves_order():
    cache = init_kv_cache(batch=1, heads=1, max_seq=4, head_dim=2)
    k1 = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
    v1 = np.array([[[[3.0, 4.0]]]], dtype=np.float32)
    k2 = np.array([[[[5.0, 6.0]]]], dtype=np.float32)
    v2 = np.array([[[[7.0, 8.0]]]], dtype=np.float32)
    append_kv(cache, k1, v1)
    append_kv(cache, k2, v2)
    k_now, v_now = current_kv(cache)
    np.testing.assert_allclose(k_now[0, 0, 0], [1.0, 2.0])
    np.testing.assert_allclose(k_now[0, 0, 1], [5.0, 6.0])
    np.testing.assert_allclose(v_now[0, 0, 0], [3.0, 4.0])
    np.testing.assert_allclose(v_now[0, 0, 1], [7.0, 8.0])

