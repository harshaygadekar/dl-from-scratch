"""Level 04: optional PyTorch pooling reference."""

import numpy as np


def max_pool2d_pytorch(x, kernel_size=2, stride=2):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for level04 verification") from exc

    x_t = torch.from_numpy(x)
    y_t = F.max_pool2d(x_t, kernel_size=kernel_size, stride=stride)
    return y_t.numpy()
