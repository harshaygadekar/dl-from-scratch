import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import init_kv_cache, append_kv
from level02_vectorized import cached_attention_step


def test_cached_attention_runs_multi_steps():
    b, h, tmax, d = 2, 4, 32, 16
    cache = init_kv_cache(b, h, tmax, d)
    for _ in range(10):
        k = np.random.randn(b, h, 1, d).astype(np.float32)
        v = np.random.randn(b, h, 1, d).astype(np.float32)
        q = np.random.randn(b, h, 1, d).astype(np.float32)
        append_kv(cache, k, v)
        out = cached_attention_step(q, cache)
        assert out.shape == (b, h, 1, d)



def test_phase_c_stress_31_cached_step_deterministic_for_fixed_cache():
    np.random.seed(31)
    b, h, tmax, d = 1, 2, 8, 4
    cache = init_kv_cache(b, h, tmax, d)
    for _ in range(3):
      append_kv(cache,
                np.random.randn(b, h, 1, d).astype(np.float32),
                np.random.randn(b, h, 1, d).astype(np.float32))
    q = np.random.randn(b, h, 1, d).astype(np.float32)
    o1 = cached_attention_step(q, cache)
    o2 = cached_attention_step(q, cache)
    np.testing.assert_allclose(o1, o2, atol=1e-6)

