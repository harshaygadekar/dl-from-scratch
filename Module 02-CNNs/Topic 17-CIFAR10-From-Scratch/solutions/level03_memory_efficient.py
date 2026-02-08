"""Level 03: scheduling helpers for CIFAR training."""

import numpy as np


def cosine_lr(base_lr, step, total_steps, min_lr=1e-4):
    if total_steps <= 0:
        return base_lr
    cosine = 0.5 * (1 + np.cos(np.pi * step / total_steps))
    return min_lr + (base_lr - min_lr) * cosine
