"""
Topic 05: MLP Forward Pass - Basic Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import MLP, Linear, relu, xavier_init, kaiming_init


class TestLinearLayer:
    """Test single linear layer."""
    
    def test_output_shape(self):
        layer = Linear(10, 5)
        x = np.random.randn(32, 10)
        out = layer.forward(x)
        assert out.shape == (32, 5)
    
    def test_single_sample(self):
        layer = Linear(10, 5)
        x = np.random.randn(10)
        out = layer.forward(x)
        assert out.shape == (5,)
    
    def test_zero_input(self):
        layer = Linear(10, 5)
        x = np.zeros((1, 10))
        out = layer.forward(x)
        np.testing.assert_allclose(out, layer.b.reshape(1, -1))


class TestMLP:
    """Test MLP forward pass."""
    
    def test_forward_shape(self):
        mlp = MLP([784, 256, 10])
        x = np.random.randn(32, 784)
        out = mlp.forward(x)
        assert out.shape == (32, 10)
    
    def test_predict_shape(self):
        mlp = MLP([784, 256, 10])
        x = np.random.randn(32, 784)
        preds = mlp.predict(x)
        assert preds.shape == (32,)
        assert all(0 <= p < 10 for p in preds)
    
    def test_num_parameters(self):
        mlp = MLP([10, 5, 3])
        expected = 10*5 + 5 + 5*3 + 3  # W1, b1, W2, b2
        assert mlp.num_parameters() == expected


class TestInitialization:
    """Test initialization schemes."""
    
    def test_xavier_variance(self):
        np.random.seed(42)
        W = xavier_init(1000, 1000)
        expected_var = 2.0 / 2000
        assert abs(W.var() - expected_var) < 0.01
    
    def test_kaiming_variance(self):
        np.random.seed(42)
        W = kaiming_init(1000, 1000)
        expected_var = 2.0 / 1000
        assert abs(W.var() - expected_var) < 0.01


class TestActivations:
    """Test activation functions."""
    
    def test_relu_positive(self):
        x = np.array([1, 2, 3])
        np.testing.assert_array_equal(relu(x), x)
    
    def test_relu_negative(self):
        x = np.array([-1, -2, -3])
        np.testing.assert_array_equal(relu(x), np.zeros(3))
    
    def test_relu_mixed(self):
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(relu(x), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
