import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import layer_norm_nchw, instance_norm_nchw, group_norm_nchw


def test_layernorm_mean_close_zero():
    x = np.random.randn(4, 8, 6, 6).astype(np.float32)
    y = layer_norm_nchw(x)
    means = y.mean(axis=(1, 2, 3))
    assert np.allclose(means, 0.0, atol=1e-5)


def test_instancenorm_mean_close_zero():
    x = np.random.randn(2, 4, 8, 8).astype(np.float32)
    y = instance_norm_nchw(x)
    means = y.mean(axis=(2, 3))
    assert np.allclose(means, 0.0, atol=1e-5)


def test_groupnorm_shape_preserved():
    x = np.random.randn(2, 8, 5, 5).astype(np.float32)
    y = group_norm_nchw(x, num_groups=4)
    assert y.shape == x.shape
