"""
Topic 08: Loss Functions - Edge Case Tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import SoftmaxCrossEntropyLoss, BCEWithLogitsLoss


class TestNumericalStability:
    def test_ce_extreme_logits(self):
        ce = SoftmaxCrossEntropyLoss()
        logits = np.array([[1000.0, 0.0, 0.0]])
        labels = np.array([[1.0, 0.0, 0.0]])
        loss = ce.forward(logits, labels)
        assert np.isfinite(loss)
    
    def test_bce_extreme_logits(self):
        bce = BCEWithLogitsLoss()
        logits = np.array([1000.0, -1000.0])
        labels = np.array([1.0, 0.0])
        loss = bce.forward(logits, labels)
        assert np.isfinite(loss)


class TestGradients:
    def test_ce_gradient_finite(self):
        ce = SoftmaxCrossEntropyLoss()
        logits = np.random.randn(16, 10)
        labels = np.eye(10)[np.random.randint(0, 10, 16)]
        ce.forward(logits, labels)
        grad = ce.backward()
        assert np.all(np.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
