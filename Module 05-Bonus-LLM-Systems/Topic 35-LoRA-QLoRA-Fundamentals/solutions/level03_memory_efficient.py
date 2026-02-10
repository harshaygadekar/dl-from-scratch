import numpy as np


def apply_lora_inplace(base_w, a, b, alpha=1.0):
    rank = a.shape[1]
    base_w += (a @ b) * (alpha / max(rank, 1))
    return base_w
