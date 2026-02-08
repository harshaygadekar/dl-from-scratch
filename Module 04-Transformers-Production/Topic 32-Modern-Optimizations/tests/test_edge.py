import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import activation_memory_bytes


def test_checkpoint_ratio_reduces_memory():
    full = activation_memory_bytes(2, 128, 256, 12, checkpoint_ratio=1.0)
    half = activation_memory_bytes(2, 128, 256, 12, checkpoint_ratio=0.5)
    assert half < full
