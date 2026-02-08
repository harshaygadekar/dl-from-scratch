"""Level 03: gradient accumulation helper."""


def accumulation_steps(effective_batch_size, micro_batch_size):
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    return max(1, effective_batch_size // micro_batch_size)
