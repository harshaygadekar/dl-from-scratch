import numpy as np


def speculative_accept_reject(target_logits, draft_tokens):
    probs = np.exp(target_logits - np.max(target_logits))
    probs /= probs.sum()
    accepted = []
    for tok in draft_tokens:
        if probs[tok] >= np.median(probs):
            accepted.append(int(tok))
        else:
            break
    return accepted
