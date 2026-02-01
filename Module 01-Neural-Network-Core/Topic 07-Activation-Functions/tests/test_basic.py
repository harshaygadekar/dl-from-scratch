"""
Topic 07: Activation Functions - Basic Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, softmax


class TestReLU:
    def test_positive_passthrough(self):
        relu = ReLU()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu.forward(x), x)
    
    def test_negative_zero(self):
        relu = ReLU()
        x = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(relu.forward(x), np.zeros(3))
    
    def test_gradient_positive(self):
        relu = ReLU()
        relu.forward(np.array([1.0, 2.0]))
        grad = relu.backward(np.array([1.0, 1.0]))
        np.testing.assert_array_equal(grad, np.array([1.0, 1.0]))
    
    def test_gradient_negative(self):
        relu = ReLU()
        relu.forward(np.array([-1.0, -2.0]))
        grad = relu.backward(np.array([1.0, 1.0]))
        np.testing.assert_array_equal(grad, np.array([0.0, 0.0]))


class TestSigmoid:
    def test_zero_equals_half(self):
        sigmoid = Sigmoid()
        assert sigmoid.forward(np.array([0.0]))[0] == pytest.approx(0.5)
    
    def test_range(self):
        sigmoid = Sigmoid()
        x = np.linspace(-10, 10, 100)
        out = sigmoid.forward(x)
        assert np.all(out > 0) and np.all(out < 1)
    
    def test_gradient_at_zero(self):
        sigmoid = Sigmoid()
        sigmoid.forward(np.array([0.0]))
        grad = sigmoid.backward(np.array([1.0]))
        assert grad[0] == pytest.approx(0.25)


class TestTanh:
    def test_zero_equals_zero(self):
        tanh = Tanh()
        assert tanh.forward(np.array([0.0]))[0] == pytest.approx(0.0)
    
    def test_range(self):
        tanh = Tanh()
        x = np.linspace(-10, 10, 100)
        out = tanh.forward(x)
        assert np.all(out >= -1) and np.all(out <= 1)


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.random.randn(5, 10)
        out = softmax(x)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(5), rtol=1e-5)
    
    def test_all_positive(self):
        x = np.random.randn(5, 10)
        out = softmax(x)
        assert np.all(out > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
