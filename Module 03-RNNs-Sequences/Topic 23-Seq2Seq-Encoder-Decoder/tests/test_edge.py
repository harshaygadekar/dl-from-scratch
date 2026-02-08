import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import truncated_length


def test_truncated_length_cap():
    assert truncated_length(100, hard_cap=32) == 32
    assert truncated_length(16, hard_cap=32) == 16
