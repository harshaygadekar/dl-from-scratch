import numpy as np


def selective_scan_naive(a, b, x):
    """Simple recurrence: h_t = a_t * h_{t-1} + b_t * x_t."""
    t = len(x)
    h = np.zeros_like(x, dtype=np.float32)
    prev = 0.0
    for i in range(t):
        prev = a[i] * prev + b[i] * x[i]
        h[i] = prev
    return h
