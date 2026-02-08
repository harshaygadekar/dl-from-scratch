import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import split_segments, checkpoint_plan, estimate_attention_flops


def test_split_segments_cover_range():
    segs = split_segments(10, 4)
    assert segs == [(0, 4), (4, 8), (8, 10)]


def test_checkpoint_plan_pattern():
    cp = checkpoint_plan(6, every_n_layers=2)
    assert cp == [0, 2, 4]


def test_flops_positive():
    assert estimate_attention_flops(128, 64) > 0
