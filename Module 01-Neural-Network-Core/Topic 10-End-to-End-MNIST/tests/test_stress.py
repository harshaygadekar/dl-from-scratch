"""Topic 10: End-to-End MNIST - Stress Tests"""

import pytest
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import MLP


class TestPerformance:
    def test_forward_speed(self):
        model = MLP([784, 256, 128, 10])
        x = np.random.randn(128, 784).astype(np.float32)
        
        start = time.time()
        for _ in range(100):
            model.forward(x)
        elapsed = time.time() - start
        
        assert elapsed < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
