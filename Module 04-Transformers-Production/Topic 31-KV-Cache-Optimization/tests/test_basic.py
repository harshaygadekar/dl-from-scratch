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
