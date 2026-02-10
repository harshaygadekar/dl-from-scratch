def speedup_estimate(serial_steps, speculative_steps):
    if speculative_steps <= 0:
        raise ValueError("speculative_steps must be positive")
    return float(serial_steps) / float(speculative_steps)
