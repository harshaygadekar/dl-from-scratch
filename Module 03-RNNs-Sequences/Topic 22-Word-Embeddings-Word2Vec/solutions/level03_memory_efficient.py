"""Level 03: token subsampling helper for frequent words."""

import numpy as np


def subsample_tokens(tokens, freqs, threshold=1e-5, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    kept = []
    for t in tokens:
        f = max(freqs.get(t, threshold), threshold)
        p_drop = max(0.0, 1.0 - np.sqrt(threshold / f))
        if rng.random() >= p_drop:
            kept.append(t)
    return kept
