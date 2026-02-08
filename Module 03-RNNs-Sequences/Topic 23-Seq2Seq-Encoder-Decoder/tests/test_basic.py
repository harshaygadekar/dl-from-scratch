import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import one_hot, encode, decode_teacher_forcing


def test_one_hot_shape():
    y = one_hot(np.array([1, 0, 2]), vocab_size=4)
    assert y.shape == (3, 4)


def test_encode_decode_shapes():
    t_src, t_tgt, b, v, h = 5, 6, 3, 8, 10
    src = one_hot(np.random.randint(0, v, size=(t_src, b)), v)
    tgt = one_hot(np.random.randint(0, v, size=(t_tgt, b)), v)

    h0 = np.zeros((b, h), dtype=np.float32)
    w_xh = np.random.randn(v, h).astype(np.float32) * 0.1
    w_hh = np.random.randn(h, h).astype(np.float32) * 0.1
    b_h = np.zeros(h, dtype=np.float32)

    enc = encode(src, h0, w_xh, w_hh, b_h)
    w_out = np.random.randn(h, v).astype(np.float32) * 0.1
    b_out = np.zeros(v, dtype=np.float32)

    logits = decode_teacher_forcing(tgt, enc, w_xh, w_hh, b_h, w_out, b_out)
    assert enc.shape == (b, h)
    assert logits.shape == (t_tgt, b, v)
