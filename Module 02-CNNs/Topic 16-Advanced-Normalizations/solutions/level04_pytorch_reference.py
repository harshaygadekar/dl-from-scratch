"""Level 04: optional PyTorch references for normalization layers."""

import numpy as np


def group_norm_pytorch_reference(x, num_groups=4):
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for level04 verification") from exc

    n, c, _, _ = x.shape
    gn = nn.GroupNorm(num_groups=num_groups, num_channels=c, affine=False)
    y = gn(torch.from_numpy(x))
    return y.numpy()
