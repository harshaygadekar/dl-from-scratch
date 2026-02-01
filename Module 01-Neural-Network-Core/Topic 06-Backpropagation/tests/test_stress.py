"""
Topic 06: Backpropagation - Stress Tests
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import MLP, one_hot


class TestPerformance:
    """Performance tests."""
    
    def test_large_batch_speed(self):
        mlp = MLP([784, 256, 128, 10])
        x = np.random.randn(1000, 784)
        y = one_hot(np.random.randint(0, 10, 1000), 10)
        
        start = time.time()
        loss = mlp.forward(x, y)
        mlp.backward()
        elapsed = time.time() - start
        
        assert elapsed < 2.0
    
    def test_many_iterations(self):
        mlp = MLP([20, 16, 8])
        x = np.random.randn(32, 20)
        y = one_hot(np.random.randint(0, 8, 32), 8)
        
        start = time.time()
        for _ in range(100):
            loss = mlp.forward(x, y)
            mlp.backward()
            mlp.update(0.01)
        elapsed = time.time() - start
        
        assert elapsed < 2.0


class TestConvergence:
    """Convergence tests."""
    
    def test_loss_decreases(self):
        np.random.seed(42)
        mlp = MLP([10, 16, 5])
        x = np.random.randn(100, 10)
        y = one_hot(np.random.randint(0, 5, 100), 5)
        
        initial_loss = mlp.forward(x, y)
        
        for _ in range(50):
            mlp.forward(x, y)
            mlp.backward()
            mlp.update(0.1)
        
        final_loss = mlp.forward(x, y)
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
