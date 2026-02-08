"""Level 04: optional PyTorch reference for depthwise/pointwise conv."""

import numpy as np


def depthwise_pytorch_reference(x, weight, stride=1, padding=0):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for level04 verification") from exc

    # PyTorch expects depthwise weights as (C_out=C_in, 1, K, K) with groups=C_in
    w = weight[:, None, :, :]
    y = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), stride=stride, padding=padding, groups=x.shape[1])
    return y.numpy()
