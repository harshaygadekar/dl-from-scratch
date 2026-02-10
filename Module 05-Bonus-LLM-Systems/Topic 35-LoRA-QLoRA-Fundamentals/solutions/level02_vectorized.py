import numpy as np


def quantize_4bit_linear(x, scale):
    q = np.round(x / scale)
    q = np.clip(q, -8, 7)
    return q.astype(np.int8)


def dequantize_4bit_linear(q, scale):
    return q.astype(np.float32) * scale
