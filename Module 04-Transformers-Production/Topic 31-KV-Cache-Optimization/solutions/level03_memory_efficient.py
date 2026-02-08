"""Level 03: circular cache index helper."""


def next_cache_index(current_len, max_seq):
    return current_len % max_seq
