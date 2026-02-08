"""Level 02: sync step over parameter dictionaries."""


def sync_sgd_step(params, worker_grads, lr):
    """params: dict[str, ndarray], worker_grads: list[dict[str, ndarray]]"""
    updated = {}
    for key in params:
        g_sum = None
        for wg in worker_grads:
            g_sum = wg[key] if g_sum is None else g_sum + wg[key]
        g_avg = g_sum / len(worker_grads)
        updated[key] = params[key] - lr * g_avg
    return updated
