import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import cosine_lr


def test_cosine_lr_monotonic_nonnegative():
    lrs = [cosine_lr(0.1, step=i, total_steps=100, min_lr=0.001) for i in range(101)]
    assert all(lr >= 0 for lr in lrs)
    assert lrs[0] > lrs[-1]
