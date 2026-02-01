"""Topic 09: Regularization - Stress Tests"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import BatchNorm1d


class TestPerformance:
    def test_batchnorm_speed(self):
        bn = BatchNorm1d(512)
        x = np.random.randn(128, 512).astype(np.float32)
        
        start = time.time()
        for _ in range(100):
            bn.forward(x)
        elapsed = time.time() - start
        
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
