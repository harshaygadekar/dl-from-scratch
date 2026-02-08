"""Level 03: activation checkpoint memory estimator."""


def activation_memory_bytes(batch, seq_len, d_model, num_layers, bytes_per_elem=4, checkpoint_ratio=1.0):
    full = batch * seq_len * d_model * num_layers * bytes_per_elem
    return int(full * checkpoint_ratio)
