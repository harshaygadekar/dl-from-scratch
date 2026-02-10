import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import next_cache_index


def test_circular_index_wraps():
    assert next_cache_index(0, 8) == 0
    assert next_cache_index(8, 8) == 0
    assert next_cache_index(9, 8) == 1



def test_phase_c_edge_31_index_progression_modulo():
    vals = [next_cache_index(i, 4) for i in range(8)]
    assert vals == [0, 1, 2, 3, 0, 1, 2, 3]

