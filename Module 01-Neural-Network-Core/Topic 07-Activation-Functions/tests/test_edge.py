"""
Topic 07: Activation Functions - Edge Case Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import Sigmoid, Tanh, softmax


class TestNumericalStability:
    def test_sigmoid_large_negative(self):
        sigmoid = Sigmoid()
        out = sigmoid.forward(np.array([-1000.0]))
        assert np.isfinite(out[0])
        assert out[0] >= 0
    
    def test_sigmoid_large_positive(self):
        sigmoid = Sigmoid()
        out = sigmoid.forward(np.array([1000.0]))
        assert np.isfinite(out[0])
        assert out[0] <= 1
    
    def test_softmax_large_values(self):
        x = np.array([[1000.0, 0.0, 0.0]])
        out = softmax(x)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out.sum(), 1.0, rtol=1e-5)


class TestEdgeCases:
    def test_empty_batch(self):
        sigmoid = Sigmoid()
        x = np.array([]).reshape(0, 5)
        out = sigmoid.forward(x)
        assert out.shape == (0, 5)
    
    def test_single_element(self):
        tanh = Tanh()
        x = np.array([[0.5]])
        out = tanh.forward(x)
        assert out.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
