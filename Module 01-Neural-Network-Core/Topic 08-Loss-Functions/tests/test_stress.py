"""
Topic 08: Loss Functions - Stress Tests
"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import MSELoss, SoftmaxCrossEntropyLoss


class TestPerformance:
    def test_mse_speed(self):
        mse = MSELoss()
        y_pred = np.random.randn(10000)
        y_true = np.random.randn(10000)
        
        start = time.time()
        for _ in range(1000):
            mse.forward(y_pred, y_true)
        elapsed = time.time() - start
        
        assert elapsed < 2.0
    
    def test_ce_speed(self):
        ce = SoftmaxCrossEntropyLoss()
        logits = np.random.randn(128, 100)
        labels = np.eye(100)[np.random.randint(0, 100, 128)]
        
        start = time.time()
        for _ in range(100):
            ce.forward(logits, labels)
        elapsed = time.time() - start
        
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
