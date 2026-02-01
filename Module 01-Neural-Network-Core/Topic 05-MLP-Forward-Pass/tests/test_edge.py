"""
Topic 05: MLP Forward Pass - Edge Case Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import MLP, Linear


class TestEdgeCases:
    """Edge case tests."""
    
    def test_single_layer(self):
        mlp = MLP([10, 5])
        x = np.random.randn(32, 10)
        out = mlp.forward(x)
        assert out.shape == (32, 5)
    
    def test_deep_network(self):
        mlp = MLP([10, 8, 8, 8, 8, 8, 5])
        x = np.random.randn(32, 10)
        out = mlp.forward(x)
        assert out.shape == (32, 5)
        assert np.all(np.isfinite(out))
    
    def test_wide_network(self):
        mlp = MLP([10, 1000, 5])
        x = np.random.randn(32, 10)
        out = mlp.forward(x)
        assert out.shape == (32, 5)
    
    def test_batch_size_one(self):
        mlp = MLP([10, 5, 3])
        x = np.random.randn(1, 10)
        out = mlp.forward(x)
        assert out.shape == (1, 3)


class TestNumericalStability:
    """Numerical stability tests."""
    
    def test_large_inputs(self):
        mlp = MLP([10, 5])
        x = np.random.randn(32, 10) * 100
        out = mlp.forward(x)
        assert np.all(np.isfinite(out))
    
    def test_small_inputs(self):
        mlp = MLP([10, 5])
        x = np.random.randn(32, 10) * 1e-10
        out = mlp.forward(x)
        assert np.all(np.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
