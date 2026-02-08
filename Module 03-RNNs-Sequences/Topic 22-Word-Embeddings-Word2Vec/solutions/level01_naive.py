"""Level 01: embedding lookup + negative sampling loss."""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class EmbeddingTable:
    def __init__(self, vocab_size, dim, scale=0.01):
        self.weight = np.random.randn(vocab_size, dim).astype(np.float32) * scale

    def lookup(self, token_ids):
        return self.weight[np.asarray(token_ids)]


def negative_sampling_loss(center_vec, pos_vec, neg_vecs):
    """Word2Vec-style negative sampling loss for one center token.

    center_vec: (D,)
    pos_vec: (D,)
    neg_vecs: (K, D)
    """
    pos_score = float(center_vec @ pos_vec)
    neg_scores = neg_vecs @ center_vec

    pos_loss = -np.log(sigmoid(pos_score) + 1e-12)
    neg_loss = -np.sum(np.log(sigmoid(-neg_scores) + 1e-12))
    return float(pos_loss + neg_loss)


def generate_skipgram_pairs(tokens, window_size=2):
    pairs = []
    for i, center in enumerate(tokens):
        left = max(0, i - window_size)
        right = min(len(tokens), i + window_size + 1)
        for j in range(left, right):
            if j != i:
                pairs.append((center, tokens[j]))
    return pairs
