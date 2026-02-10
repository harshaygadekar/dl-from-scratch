import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import decoder_autoregressive_step


def test_autoregressive_step_runs():
    b, t_dec, t_enc, d, h = 4, 64, 64, 32, 64
    prefix = np.random.randn(b, t_dec, d).astype(np.float32)
    enc = np.random.randn(b, t_enc, d).astype(np.float32)
    p = {
        "w_qs": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_ks": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_vs": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_os": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_qc": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_kc": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_vc": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_oc": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w1": np.random.randn(d, h).astype(np.float32) * 0.05,
        "b1": np.zeros(h, dtype=np.float32),
        "w2": np.random.randn(h, d).astype(np.float32) * 0.05,
        "b2": np.zeros(d, dtype=np.float32),
    }
    y = decoder_autoregressive_step(prefix, enc, p)
    assert y.shape == (b, d)


from level01_naive import decoder_block_forward


def test_phase_c_regression_29_autoregressive_matches_last_token():
    np.random.seed(19)
    b, t_dec, t_enc, d, h = 2, 9, 7, 8, 16
    prefix = np.random.randn(b, t_dec, d).astype(np.float32)
    enc = np.random.randn(b, t_enc, d).astype(np.float32)
    p = {
        "w_qs": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_ks": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_vs": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_os": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_qc": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_kc": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_vc": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w_oc": np.random.randn(d, d).astype(np.float32) * 0.05,
        "w1": np.random.randn(d, h).astype(np.float32) * 0.05,
        "b1": np.zeros(h, dtype=np.float32),
        "w2": np.random.randn(h, d).astype(np.float32) * 0.05,
        "b2": np.zeros(d, dtype=np.float32),
    }
    full = decoder_block_forward(prefix, enc, p)
    step = decoder_autoregressive_step(prefix, enc, p)
    np.testing.assert_allclose(step, full[:, -1, :], atol=1e-6)

