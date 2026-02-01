"""
Topic 06: Backpropagation - Basic Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import Linear, ReLU, MLP, one_hot


class TestLinearBackward:
    """Test linear layer backward pass."""
    
    def test_weight_gradient_shape(self):
        layer = Linear(10, 5)
        x = np.random.randn(8, 10)
        out = layer.forward(x)
        grad_out = np.ones_like(out)
        layer.backward(grad_out)
        
        assert layer.grad_W.shape == (10, 5)
        assert layer.grad_b.shape == (5,)
    
    def test_input_gradient_shape(self):
        layer = Linear(10, 5)
        x = np.random.randn(8, 10)
        out = layer.forward(x)
        grad_out = np.ones_like(out)
        grad_input = layer.backward(grad_out)
        
        assert grad_input.shape == x.shape
    
    def test_gradient_direction(self):
        """Gradient should point in direction that reduces loss."""
        layer = Linear(2, 1)
        layer.W = np.array([[1.0], [1.0]])
        layer.b = np.array([0.0])
        
        x = np.array([[1.0, 1.0]])
        out = layer.forward(x)
        
        # Positive grad_output means loss increases with output
        grad_out = np.array([[1.0]])
        layer.backward(grad_out)
        
        # To decrease loss, we should decrease weights
        # So gradient should be positive (subtracting it decreases weights)
        assert layer.grad_W[0, 0] > 0


class TestReLUBackward:
    """Test ReLU backward pass."""
    
    def test_relu_pass_positive(self):
        """Gradient passes through for positive inputs."""
        relu = ReLU()
        x = np.array([[1.0, 2.0, 3.0]])
        relu.forward(x)
        grad_out = np.array([[1.0, 1.0, 1.0]])
        grad_in = relu.backward(grad_out)
        
        np.testing.assert_array_equal(grad_in, grad_out)
    
    def test_relu_block_negative(self):
        """Gradient is blocked for negative inputs."""
        relu = ReLU()
        x = np.array([[-1.0, -2.0, -3.0]])
        relu.forward(x)
        grad_out = np.array([[1.0, 1.0, 1.0]])
        grad_in = relu.backward(grad_out)
        
        np.testing.assert_array_equal(grad_in, np.zeros_like(grad_out))
    
    def test_relu_mixed(self):
        relu = ReLU()
        x = np.array([[-1.0, 0.5, -0.5, 2.0]])
        relu.forward(x)
        grad_out = np.array([[1.0, 2.0, 3.0, 4.0]])
        grad_in = relu.backward(grad_out)
        
        expected = np.array([[0.0, 2.0, 0.0, 4.0]])
        np.testing.assert_array_equal(grad_in, expected)


class TestMLPBackward:
    """Test full MLP backward pass."""
    
    def test_backward_runs(self):
        mlp = MLP([10, 8, 5])
        x = np.random.randn(4, 10)
        y = one_hot([0, 1, 2, 3], 5)
        
        loss = mlp.forward(x, y)
        mlp.backward()  # Should not raise
        
        assert mlp.layers[0].grad_W is not None
        assert mlp.layers[1].grad_W is not None
    
    def test_gradient_update_changes_loss(self):
        np.random.seed(42)
        mlp = MLP([5, 4, 3])
        x = np.random.randn(10, 5)
        y = one_hot(np.random.randint(0, 3, 10), 3)
        
        loss1 = mlp.forward(x, y)
        mlp.backward()
        mlp.update(lr=0.1)
        loss2 = mlp.forward(x, y)
        
        # Loss should decrease after gradient step
        assert loss2 < loss1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
