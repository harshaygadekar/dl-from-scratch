import numpy as np


def route_with_capacity(top1_idx, num_experts, capacity):
    assignments = -np.ones_like(top1_idx)
    counts = np.zeros(num_experts, dtype=np.int64)
    for i, e in enumerate(top1_idx):
        if counts[e] < capacity:
            assignments[i] = e
            counts[e] += 1
    return assignments, counts
