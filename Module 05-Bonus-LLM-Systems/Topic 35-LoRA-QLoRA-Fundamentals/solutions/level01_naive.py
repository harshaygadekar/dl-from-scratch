import numpy as np


def lora_delta(a, b, alpha=1.0):
    rank = a.shape[1]
    return (a @ b) * (alpha / max(rank, 1))


def apply_lora(base_w, a, b, alpha=1.0):
    return base_w + lora_delta(a, b, alpha=alpha)
