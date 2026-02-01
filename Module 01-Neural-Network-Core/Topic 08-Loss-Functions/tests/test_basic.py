"""
Topic 08: Loss Functions - Basic Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import MSELoss, MAELoss, SoftmaxCrossEntropyLoss, BCEWithLogitsLoss


class TestMSE:
    def test_zero_error(self):
        mse = MSELoss()
        y = np.array([1.0, 2.0, 3.0])
        assert mse.forward(y, y) == pytest.approx(0.0)
    
    def test_simple(self):
        mse = MSELoss()
        y_pred = np.array([2.0, 3.0])
        y_true = np.array([1.0, 1.0])
        # (1^2 + 2^2) / 2 = 2.5
        assert mse.forward(y_pred, y_true) == pytest.approx(2.5)


class TestMAE:
    def test_zero_error(self):
        mae = MAELoss()
        y = np.array([1.0, 2.0, 3.0])
        assert mae.forward(y, y) == pytest.approx(0.0)
    
    def test_simple(self):
        mae = MAELoss()
        y_pred = np.array([2.0, 4.0])
        y_true = np.array([1.0, 1.0])
        # (1 + 3) / 2 = 2
        assert mae.forward(y_pred, y_true) == pytest.approx(2.0)


class TestCrossEntropy:
    def test_perfect_prediction(self):
        ce = SoftmaxCrossEntropyLoss()
        logits = np.array([[10.0, 0.0, 0.0]])
        labels = np.array([[1.0, 0.0, 0.0]])
        loss = ce.forward(logits, labels)
        assert loss < 0.01
    
    def test_bad_prediction(self):
        ce = SoftmaxCrossEntropyLoss()
        logits = np.array([[0.0, 10.0, 0.0]])
        labels = np.array([[1.0, 0.0, 0.0]])
        loss = ce.forward(logits, labels)
        assert loss > 1.0


class TestBCE:
    def test_perfect_prediction(self):
        bce = BCEWithLogitsLoss()
        logits = np.array([10.0])  # Strong positive
        labels = np.array([1.0])
        loss = bce.forward(logits, labels)
        assert loss < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
