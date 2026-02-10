import numpy as np


def load_balance_penalty(counts):
    counts = counts.astype(np.float32)
    target = counts.mean()
    return float(np.mean((counts - target) ** 2))
