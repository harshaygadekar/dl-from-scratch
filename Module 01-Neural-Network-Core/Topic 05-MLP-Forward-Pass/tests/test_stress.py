"""
Topic 05: MLP Forward Pass - Stress Tests
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import MLP


class TestPerformance:
    """Performance tests."""
    
    def test_large_batch(self):
        mlp = MLP([784, 256, 128, 10])
        x = np.random.randn(1000, 784)
        
        start = time.time()
        out = mlp.forward(x)
        elapsed = time.time() - start
        
        assert out.shape == (1000, 10)
        assert elapsed < 1.0
    
    def test_many_forward_passes(self):
        mlp = MLP([100, 50, 10])
        x = np.random.randn(32, 100)
        
        start = time.time()
        for _ in range(100):
            out = mlp.forward(x)
        elapsed = time.time() - start
        
        assert elapsed < 1.0


class TestActivationHealth:
    """Test activation statistics."""
    
    def test_activation_variance_stable(self):
        """Check that activations don't vanish or explode."""
        np.random.seed(42)
        mlp = MLP([100, 100, 100, 100, 100, 10], init='kaiming')
        x = np.random.randn(100, 100)
        
        out = mlp.forward(x)
        
        # Output should have reasonable variance
        assert 0.001 < out.var() < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
