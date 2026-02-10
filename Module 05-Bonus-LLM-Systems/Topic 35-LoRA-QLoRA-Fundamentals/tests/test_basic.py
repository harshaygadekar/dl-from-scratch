import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level01_naive import lora_delta, apply_lora


def test_lora_delta_shape():
    a = np.random.randn(8, 2).astype(np.float32)
    b = np.random.randn(2, 6).astype(np.float32)
    d = lora_delta(a, b, alpha=2.0)
    assert d.shape == (8, 6)


def test_apply_lora_changes_weights():
    w = np.zeros((8, 6), dtype=np.float32)
    a = np.random.randn(8, 2).astype(np.float32)
    b = np.random.randn(2, 6).astype(np.float32)
    out = apply_lora(w, a, b, alpha=1.0)
    assert np.any(np.abs(out) > 0)
