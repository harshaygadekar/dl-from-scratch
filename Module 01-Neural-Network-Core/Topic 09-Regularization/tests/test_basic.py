"""Topic 09: Regularization - Basic Tests"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import Dropout, BatchNorm1d


class TestDropout:
    def test_training_mode(self):
        drop = Dropout(p=0.5)
        x = np.ones((100, 100))
        out = drop.forward(x)
        zero_rate = np.mean(out == 0)
        assert 0.4 < zero_rate < 0.6
    
    def test_eval_mode(self):
        drop = Dropout(p=0.5)
        drop.training = False
        x = np.ones((10, 10))
        out = drop.forward(x)
        assert np.allclose(out, x)


class TestBatchNorm:
    def test_normalization(self):
        bn = BatchNorm1d(64)
        x = np.random.randn(32, 64) * 5 + 10
        out = bn.forward(x)
        assert abs(out.mean()) < 0.1
        assert abs(out.std() - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
