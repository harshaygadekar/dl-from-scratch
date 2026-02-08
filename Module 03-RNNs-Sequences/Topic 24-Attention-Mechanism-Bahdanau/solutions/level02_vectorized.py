"""Level 02: batched decoder-attention wrapper."""

import numpy as np
from level01_naive import bahdanau_context


def attend_decoder_states(decoder_states, keys, values, w_q, w_k, v_a, b_a, mask=None):
    """decoder_states: (T_dec, B, H_dec) -> contexts: (T_dec, B, H_val)."""
    contexts = []
    weights_all = []
    for t in range(decoder_states.shape[0]):
        c_t, a_t = bahdanau_context(decoder_states[t], keys, values, w_q, w_k, v_a, b_a, mask=mask)
        contexts.append(c_t)
        weights_all.append(a_t)

    return np.stack(contexts, axis=0), np.stack(weights_all, axis=0)
