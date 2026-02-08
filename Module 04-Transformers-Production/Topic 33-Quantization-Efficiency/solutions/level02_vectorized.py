"""Level 02: per-channel quantization helpers."""

import numpy as np


def per_channel_scales(w, axis=0, eps=1e-8):
    max_abs = np.max(np.abs(w), axis=axis, keepdims=True)
    return np.maximum(max_abs / 127.0, eps)


def quantize_per_channel(w, axis=0):
    scales = per_channel_scales(w, axis=axis)
    q = np.round(w / scales)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scales
