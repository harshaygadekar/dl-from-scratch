"""Level 01: KV cache append/read primitives."""

import numpy as np


def init_kv_cache(batch, heads, max_seq, head_dim, dtype=np.float32):
    k = np.zeros((batch, heads, max_seq, head_dim), dtype=dtype)
    v = np.zeros((batch, heads, max_seq, head_dim), dtype=dtype)
    return {"k": k, "v": v, "len": 0}


def append_kv(cache, k_new, v_new):
    """Append one step.

    k_new/v_new: (B, H, 1, D)
    """
    idx = cache["len"]
    cache["k"][:, :, idx:idx + 1, :] = k_new
    cache["v"][:, :, idx:idx + 1, :] = v_new
    cache["len"] += 1
    return cache


def current_kv(cache):
    l = cache["len"]
    return cache["k"][:, :, :l, :], cache["v"][:, :, :l, :]
