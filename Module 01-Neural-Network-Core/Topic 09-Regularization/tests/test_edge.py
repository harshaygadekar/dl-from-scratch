"""Topic 09: Regularization - Edge Tests"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import BatchNorm1d, Dropout


class TestBatchNormEdge:
    def test_running_stats(self):
        bn = BatchNorm1d(10)
        for _ in range(100):
            x = np.random.randn(16, 10) * 2 + 5
            bn.forward(x)
        
        assert np.abs(bn.running_mean - 5).max() < 1.0
        assert np.abs(bn.running_var - 4).max() < 1.0
    
    def test_eval_mode(self):
        bn = BatchNorm1d(10)
        x = np.random.randn(16, 10)
        bn.forward(x)
        bn.training = False
        out = bn.forward(x)
        assert np.all(np.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
