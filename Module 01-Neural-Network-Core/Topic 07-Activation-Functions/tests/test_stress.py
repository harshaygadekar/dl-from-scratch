"""
Topic 07: Activation Functions - Stress Tests
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import ReLU, Sigmoid, Tanh, softmax


class TestPerformance:
    def test_relu_speed(self):
        relu = ReLU()
        x = np.random.randn(1000, 1000)
        
        start = time.time()
        for _ in range(100):
            relu.forward(x)
        elapsed = time.time() - start
        
        assert elapsed < 2.0
    
    def test_softmax_speed(self):
        x = np.random.randn(1000, 100)
        
        start = time.time()
        for _ in range(100):
            softmax(x)
        elapsed = time.time() - start
        
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
