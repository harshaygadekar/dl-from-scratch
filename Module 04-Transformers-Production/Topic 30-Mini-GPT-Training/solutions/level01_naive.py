"""Level 01: minimal char-level language modeling scaffold."""

import numpy as np


def build_char_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_text(text, stoi):
    return np.array([stoi[ch] for ch in text], dtype=np.int64)


def make_batches(tokens, block_size=16, batch_size=8):
    if len(tokens) <= block_size:
        raise ValueError("token sequence too short for given block_size")

    x_batches = []
    y_batches = []
    for _ in range(batch_size):
        i = np.random.randint(0, len(tokens) - block_size - 1)
        x = tokens[i:i + block_size]
        y = tokens[i + 1:i + block_size + 1]
        x_batches.append(x)
        y_batches.append(y)
    return np.stack(x_batches), np.stack(y_batches)


def cross_entropy_from_logits(logits, targets):
    """logits: (B, T, V), targets: (B, T)."""
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    b, t = targets.shape
    gather = probs[np.arange(b)[:, None], np.arange(t)[None, :], targets]
    return float(-np.mean(np.log(gather + 1e-12)))
