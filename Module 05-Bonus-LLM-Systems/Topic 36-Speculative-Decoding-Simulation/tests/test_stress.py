import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level03_memory_efficient import speedup_estimate


def test_speedup_estimate_positive():
    s = speedup_estimate(serial_steps=1024, speculative_steps=256)
    assert s > 1.0
