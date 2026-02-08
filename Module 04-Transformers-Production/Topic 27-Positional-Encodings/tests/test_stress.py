import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import PositionalEncodingCache


def test_cache_reuses_arrays():
    cache = PositionalEncodingCache()
    a = cache.get(128, 64)
    b = cache.get(128, 64)
    assert a is b
