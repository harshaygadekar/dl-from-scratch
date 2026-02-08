"""Level 03: memory footprint estimator for quantized tensors."""


def tensor_memory_bytes(num_elements, bits):
    return int(num_elements * bits / 8)
