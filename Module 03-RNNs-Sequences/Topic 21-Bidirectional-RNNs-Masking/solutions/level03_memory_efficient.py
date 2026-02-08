"""Level 03: trim padded sequence tails before compute."""

import numpy as np


def trim_to_max_length(x_batch_first, lengths):
    max_len = int(np.max(lengths))
    return x_batch_first[:, :max_len]
