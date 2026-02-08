import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import allreduce_volume_bytes


def test_allreduce_volume_positive():
    vol = allreduce_volume_bytes(num_params=10_000_000, bytes_per_param=2, world_size=8)
    assert vol > 0
