"""Level 03: positional cache helper."""

from level01_naive import sinusoidal_positional_encoding


class PositionalEncodingCache:
    def __init__(self):
        self._cache = {}

    def get(self, seq_len, d_model):
        key = (seq_len, d_model)
        if key not in self._cache:
            self._cache[key] = sinusoidal_positional_encoding(seq_len, d_model)
        return self._cache[key]
