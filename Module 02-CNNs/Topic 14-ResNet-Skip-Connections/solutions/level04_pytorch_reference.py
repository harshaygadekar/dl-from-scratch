"""Level 04: optional PyTorch residual reference."""

import numpy as np


def residual_block_pytorch_reference(x):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for level04 verification") from exc

    n, c, h, w = x.shape
    block = nn.Sequential(
        nn.Conv2d(c, c, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(c, c, kernel_size=1),
    )

    x_t = torch.from_numpy(x)
    y = torch.relu(block(x_t) + x_t)
    return y.detach().numpy()
