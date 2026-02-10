import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import scaled_dot_product_attention, causal_mask


def test_causal_mask_blocks_future_attention():
    b, t, d = 1, 4, 4
    x = np.ones((b, t, d), dtype=np.float32)
    out, w = scaled_dot_product_attention(x, x, x, mask=causal_mask(t))
    assert np.all(w[0, 0, 1:] < 1e-6)



def test_phase_c_edge_25_full_causal_triangle():
    t = 6
    x = np.ones((1, t, 4), dtype=np.float32)
    _, w = scaled_dot_product_attention(x, x, x, mask=causal_mask(t))
    upper = np.triu(w[0], k=1)
    assert np.all(upper < 1e-6)

