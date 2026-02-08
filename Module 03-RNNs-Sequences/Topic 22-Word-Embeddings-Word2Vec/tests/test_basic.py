import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level01_naive import EmbeddingTable, generate_skipgram_pairs, negative_sampling_loss


def test_embedding_lookup_shape():
    emb = EmbeddingTable(vocab_size=100, dim=16)
    vecs = emb.lookup([1, 2, 3])
    assert vecs.shape == (3, 16)


def test_skipgram_pairs_nonempty():
    tokens = [1, 2, 3, 4]
    pairs = generate_skipgram_pairs(tokens, window_size=1)
    assert len(pairs) > 0


def test_negative_sampling_loss_finite():
    center = np.random.randn(8).astype(np.float32)
    pos = np.random.randn(8).astype(np.float32)
    neg = np.random.randn(5, 8).astype(np.float32)
    loss = negative_sampling_loss(center, pos, neg)
    assert np.isfinite(loss)
