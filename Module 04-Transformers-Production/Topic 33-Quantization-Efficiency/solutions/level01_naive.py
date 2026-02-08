"""Level 01: int8 quantization primitives."""

import numpy as np


def symmetric_quant_params(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    max_abs = float(np.max(np.abs(x)))
    scale = max(max_abs / qmax, 1e-8)
    return scale


def quantize_int8(x, scale):
    q = np.round(x / scale)
    q = np.clip(q, -127, 127)
    return q.astype(np.int8)


def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale


def fake_quantize(x, scale):
    return dequantize_int8(quantize_int8(x, scale), scale)
