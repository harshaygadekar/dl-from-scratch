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



def test_phase_c_edge_27_rope_preserves_pair_norm():
    np.random.seed(3)
    x = np.random.randn(2, 5, 16).astype(np.float32)
    y = apply_rope(x)
    half = x.shape[-1] // 2
    x_norm = np.sum(x[..., :half] ** 2 + x[..., half:] ** 2, axis=-1)
    y_norm = np.sum(y[..., :half] ** 2 + y[..., half:] ** 2, axis=-1)
    np.testing.assert_allclose(x_norm, y_norm, rtol=1e-5, atol=1e-5)

