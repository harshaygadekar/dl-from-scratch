"""Level 01: data-parallel splitting and gradient aggregation logic."""

import numpy as np


def split_batch(x, num_workers):
    """Split first dimension across workers."""
    return np.array_split(x, num_workers, axis=0)


def average_gradients(worker_grads):
    """Average a list of gradient arrays with identical shape."""
    stacked = np.stack(worker_grads, axis=0)
    return np.mean(stacked, axis=0)


def sgd_update(param, grad, lr):
    return param - lr * grad
