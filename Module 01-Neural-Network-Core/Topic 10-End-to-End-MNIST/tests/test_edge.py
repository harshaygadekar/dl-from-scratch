"""Topic 10: End-to-End MNIST - Edge Tests"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import MLP, SoftmaxCrossEntropy


class TestEdgeCases:
    def test_single_sample(self):
        model = MLP([10, 5, 3])
        x = np.random.randn(1, 10)
        out = model.forward(x)
        assert out.shape == (1, 3)
    
    def test_loss_gradient(self):
        loss_fn = SoftmaxCrossEntropy()
        logits = np.array([[1.0, 2.0, 3.0]])
        y = np.array([[0.0, 0.0, 1.0]])
        loss_fn.forward(logits, y)
        grad = loss_fn.backward()
        assert np.all(np.isfinite(grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
