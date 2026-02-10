import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))
from level03_memory_efficient import apply_lora_inplace


def test_inplace_large_tensor_runs():
    w = np.zeros((512, 512), dtype=np.float32)
    a = np.random.randn(512, 8).astype(np.float32)
    b = np.random.randn(8, 512).astype(np.float32)
    out = apply_lora_inplace(w, a, b, alpha=1.0)
    assert out.shape == (512, 512)
