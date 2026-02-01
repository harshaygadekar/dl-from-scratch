"""
Topic 04: Single Layer Perceptron - Stress Tests
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import Perceptron


class TestPerformance:
    """Performance tests."""
    
    def test_large_batch_performance(self):
        """Should handle large batches efficiently."""
        X = np.random.randn(10000, 100)
        y = np.random.randint(0, 2, 10000).astype(float)
        model = Perceptron(100)
        
        start = time.time()
        y_pred = model.forward(X)
        model.backward(X, y, y_pred)
        model.update(0.1)
        elapsed = time.time() - start
        
        assert elapsed < 1.0


class TestConvergence:
    """Convergence tests."""
    
    def test_convergence_easy_problem(self):
        """Should converge on well-separated data."""
        np.random.seed(42)
        X_pos = np.random.randn(100, 2) * 0.1 + np.array([5, 5])
        X_neg = np.random.randn(100, 2) * 0.1 + np.array([-5, -5])
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 100 + [0] * 100, dtype=float)
        
        model = Perceptron(2)
        for _ in range(50):
            y_pred = model.forward(X)
            model.backward(X, y, y_pred)
            model.update(0.5)
        
        assert model.accuracy(X, y) > 0.99
    
    def test_loss_decreases(self):
        """Loss should decrease during training."""
        np.random.seed(42)
        X_pos = np.random.randn(100, 2) + np.array([2, 2])
        X_neg = np.random.randn(100, 2) + np.array([-2, -2])
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * 100 + [0] * 100, dtype=float)
        
        model = Perceptron(2)
        initial_loss = model.compute_loss(model.forward(X), y)
        
        for _ in range(50):
            y_pred = model.forward(X)
            model.backward(X, y, y_pred)
            model.update(0.3)
        
        final_loss = model.compute_loss(model.forward(X), y)
        assert final_loss < initial_loss


class TestStability:
    """Stability tests."""
    
    def test_many_epochs(self):
        """Should remain stable over many epochs."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randint(0, 2, 100).astype(float)
        model = Perceptron(2)
        
        for _ in range(1000):
            y_pred = model.forward(X)
            model.backward(X, y, y_pred)
            model.update(0.01)
        
        assert np.all(np.isfinite(model.w))
        assert np.isfinite(model.b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
