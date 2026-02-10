import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import build_char_vocab, encode_text, make_batches, cross_entropy_from_logits
from level02_vectorized import init_bigram_logits, bigram_forward


def test_vocab_encode_roundtrip_len():
    text = "hello"
    stoi, _ = build_char_vocab(text)
    tokens = encode_text(text, stoi)
    assert len(tokens) == len(text)


def test_batch_shapes():
    tokens = np.arange(100, dtype=np.int64)
    x, y = make_batches(tokens, block_size=8, batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_bigram_forward_shape_and_loss_finite():
    b, t, v = 3, 6, 20
    inp = np.random.randint(0, v, size=(b, t))
    table = init_bigram_logits(v)
    logits = bigram_forward(inp, table)
    loss = cross_entropy_from_logits(logits, inp)
    assert logits.shape == (b, t, v)
    assert np.isfinite(loss)


from level02_vectorized import generate_bigram


def test_phase_c_basic_30_generation_length_and_range():
    np.random.seed(23)
    vocab = 12
    table = init_bigram_logits(vocab)
    seq = generate_bigram(start_token=3, logits_table=table, max_new_tokens=15)
    assert len(seq) == 16
    assert min(seq) >= 0 and max(seq) < vocab

