"""Level 03: truncated decode helper."""


def truncated_length(max_steps, hard_cap=64):
    return min(max_steps, hard_cap)
