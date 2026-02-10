import numpy as np


def greedy_next(logits):
    return int(np.argmax(logits))


def propose_draft(draft_logits, k=4):
    order = np.argsort(draft_logits)[::-1]
    return order[:k].astype(np.int64)
