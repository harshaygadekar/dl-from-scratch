"""Level 02: wrapper for batched causal attention."""

from level01_naive import causal_mask, scaled_dot_product_attention


def causal_self_attention(x):
    """x: (B, T, D), using x as q/k/v for a minimal self-attn block."""
    t = x.shape[1]
    mask = causal_mask(t)
    return scaled_dot_product_attention(x, x, x, mask=mask)
