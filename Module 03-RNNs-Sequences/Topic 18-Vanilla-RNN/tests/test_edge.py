import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import clip_grad_norm


def test_clip_grad_no_change_when_small():
    g = np.array([0.1, 0.2], dtype=np.float32)
    out = clip_grad_norm(g, max_norm=1.0)
    np.testing.assert_allclose(out, g)


def test_clip_grad_reduces_norm():
    g = np.array([3.0, 4.0], dtype=np.float32)
    out = clip_grad_norm(g, max_norm=1.0)
    assert np.linalg.norm(out) <= 1.000001
