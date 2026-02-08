import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import accumulation_steps


def test_accumulation_steps_nonzero():
    assert accumulation_steps(256, 32) == 8
