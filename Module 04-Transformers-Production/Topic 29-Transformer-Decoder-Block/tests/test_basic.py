import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import decoder_block_forward


def test_decoder_block_shape():
    b, t_dec, t_enc, d, h = 2, 5, 7, 16, 32
    x = np.random.randn(b, t_dec, d).astype(np.float32)
    enc = np.random.randn(b, t_enc, d).astype(np.float32)
    p = {
        "w_qs": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_ks": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_vs": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_os": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_qc": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_kc": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_vc": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_oc": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w1": np.random.randn(d, h).astype(np.float32) * 0.1,
        "b1": np.zeros(h, dtype=np.float32),
        "w2": np.random.randn(h, d).astype(np.float32) * 0.1,
        "b2": np.zeros(d, dtype=np.float32),
    }
    y = decoder_block_forward(x, enc, p)
    assert y.shape == x.shape



def test_phase_c_basic_29_decoder_block_finite_outputs():
    b, t_dec, t_enc, d, h = 1, 4, 4, 8, 16
    np.random.seed(17)
    x = np.random.randn(b, t_dec, d).astype(np.float32)
    enc = np.random.randn(b, t_enc, d).astype(np.float32)
    p = {
        "w_qs": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_ks": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_vs": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_os": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_qc": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_kc": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_vc": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w_oc": np.random.randn(d, d).astype(np.float32) * 0.1,
        "w1": np.random.randn(d, h).astype(np.float32) * 0.1,
        "b1": np.zeros(h, dtype=np.float32),
        "w2": np.random.randn(h, d).astype(np.float32) * 0.1,
        "b2": np.zeros(d, dtype=np.float32),
    }
    y = decoder_block_forward(x, enc, p)
    assert np.isfinite(y).all()

