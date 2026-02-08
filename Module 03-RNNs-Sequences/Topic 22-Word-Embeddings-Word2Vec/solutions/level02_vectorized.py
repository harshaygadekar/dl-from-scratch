"""Level 02: batched negative sampling utility."""

import numpy as np
from level01_naive import sigmoid


def negative_sampling_loss_batch(center_vecs, pos_vecs, neg_vecs):
    """Batched variant.

    center_vecs: (B, D)
    pos_vecs: (B, D)
    neg_vecs: (B, K, D)
    """
    pos_scores = np.sum(center_vecs * pos_vecs, axis=1)
    neg_scores = np.einsum("bkd,bd->bk", neg_vecs, center_vecs)

    pos_loss = -np.log(sigmoid(pos_scores) + 1e-12)
    neg_loss = -np.sum(np.log(sigmoid(-neg_scores) + 1e-12), axis=1)
    return np.mean(pos_loss + neg_loss)
