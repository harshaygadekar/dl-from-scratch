"""Level 03: communication volume estimator."""


def allreduce_volume_bytes(num_params, bytes_per_param, world_size):
    """Approximate ring all-reduce volume per step.

    Formula (rough): 2 * (world_size - 1) / world_size * payload
    """
    payload = num_params * bytes_per_param
    return int(2 * (world_size - 1) / world_size * payload)
