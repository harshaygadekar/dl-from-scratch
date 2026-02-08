import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level02_vectorized import attend_decoder_states


def test_attention_batch_runs():
    t_enc, t_dec, b, h_enc, h_dec, a = 64, 48, 8, 32, 32, 16
    dec = np.random.randn(t_dec, b, h_dec).astype(np.float32)
    keys = np.random.randn(t_enc, b, h_enc).astype(np.float32)
    values = np.random.randn(t_enc, b, h_enc).astype(np.float32)
    w_q = np.random.randn(h_dec, a).astype(np.float32) * 0.1
    w_k = np.random.randn(h_enc, a).astype(np.float32) * 0.1
    v_a = np.random.randn(a).astype(np.float32) * 0.1
    b_a = np.zeros(a, dtype=np.float32)

    context, attn = attend_decoder_states(dec, keys, values, w_q, w_k, v_a, b_a)
    assert context.shape == (t_dec, b, h_enc)
    assert attn.shape == (t_dec, b, t_enc)
