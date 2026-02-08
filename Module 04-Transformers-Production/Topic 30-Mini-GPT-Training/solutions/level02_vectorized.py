"""Level 02: simple bigram baseline and generation."""

import numpy as np


def init_bigram_logits(vocab_size, scale=0.01):
    return np.random.randn(vocab_size, vocab_size).astype(np.float32) * scale


def bigram_forward(input_tokens, logits_table):
    """input_tokens: (B, T) -> logits: (B, T, V)"""
    return logits_table[input_tokens]


def sample_next(logits_row, temperature=1.0):
    x = logits_row / max(temperature, 1e-6)
    x = x - np.max(x)
    p = np.exp(x)
    p = p / np.sum(p)
    return int(np.random.choice(len(p), p=p))


def generate_bigram(start_token, logits_table, max_new_tokens=20):
    tokens = [int(start_token)]
    for _ in range(max_new_tokens):
        nxt = sample_next(logits_table[tokens[-1]])
        tokens.append(nxt)
    return tokens
