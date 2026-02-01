"""Topic 10: End-to-End MNIST - Basic Tests"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import MLP, Linear, ReLU, SoftmaxCrossEntropy


class TestComponents:
    def test_linear_shapes(self):
        layer = Linear(784, 256)
        x = np.random.randn(32, 784)
        out = layer.forward(x)
        assert out.shape == (32, 256)
    
    def test_relu(self):
        relu = ReLU()
        x = np.array([-1, 0, 1])
        out = relu.forward(x)
        assert np.allclose(out, [0, 0, 1])


class TestMLP:
    def test_forward(self):
        model = MLP([784, 128, 10])
        x = np.random.randn(16, 784)
        out = model.forward(x)
        assert out.shape == (16, 10)
    
    def test_backward(self):
        model = MLP([10, 5, 3])
        x = np.random.randn(4, 10)
        model.forward(x)
        grad = np.random.randn(4, 3)
        model.backward(grad)  # Should not error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
