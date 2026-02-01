"""
Topic 06: Backpropagation - Edge Case Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import Linear, ReLU, MLP, one_hot


class TestNumericalGradient:
    """Test gradients against numerical approximation."""
    
    def numerical_gradient(self, f, x, eps=1e-5):
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy()
            x_plus.flat[i] += eps
            x_minus = x.copy()
            x_minus.flat[i] -= eps
            grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad
    
    def test_linear_weight_gradient(self):
        np.random.seed(42)
        layer = Linear(3, 2)
        x = np.random.randn(4, 3).astype(np.float64)
        
        def loss_fn(W_flat):
            W = W_flat.reshape(layer.W.shape)
            return np.sum((x @ W + layer.b) ** 2)
        
        out = layer.forward(x)
        layer.backward(2 * out)  # d/dy of sum(y^2)
        
        numerical = self.numerical_gradient(loss_fn, layer.W.flatten())
        analytical = layer.grad_W.flatten()
        
        rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
        assert rel_error.max() < 1e-4


class TestEdgeCases:
    """Edge case tests."""
    
    def test_batch_size_one(self):
        mlp = MLP([5, 3, 2])
        x = np.random.randn(1, 5)
        y = one_hot([0], 2)
        
        loss = mlp.forward(x, y)
        mlp.backward()
        
        assert np.all(np.isfinite(mlp.layers[0].grad_W))
    
    def test_all_zeros_input(self):
        mlp = MLP([5, 3, 2])
        x = np.zeros((4, 5))
        y = one_hot([0, 1, 0, 1], 2)
        
        loss = mlp.forward(x, y)
        mlp.backward()
        
        assert np.isfinite(loss)
    
    def test_deep_network(self):
        mlp = MLP([10, 8, 8, 8, 8, 5])
        x = np.random.randn(4, 10)
        y = one_hot([0, 1, 2, 3], 5)
        
        loss = mlp.forward(x, y)
        mlp.backward()
        
        # Gradients should still be finite
        for layer in mlp.layers:
            assert np.all(np.isfinite(layer.grad_W))
            assert np.all(np.isfinite(layer.grad_b))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
