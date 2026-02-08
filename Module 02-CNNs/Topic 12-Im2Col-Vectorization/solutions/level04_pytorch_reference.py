"""Level 04: optional PyTorch reference for im2col-based conv outputs."""

import numpy as np


def conv2d_pytorch_reference(x, weight, bias=None, stride=1, padding=0):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for level04 verification") from exc

    x_t = torch.from_numpy(x)
    w_t = torch.from_numpy(weight)
    b_t = torch.from_numpy(bias) if bias is not None else None
    y_t = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding)
    return y_t.numpy()


if __name__ == "__main__":
    print("Use this file only for optional reference checks.")
