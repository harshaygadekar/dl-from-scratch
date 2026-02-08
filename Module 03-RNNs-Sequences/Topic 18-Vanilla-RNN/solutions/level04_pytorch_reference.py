"""Level 04: optional PyTorch RNN reference."""

import numpy as np


def rnn_reference(x, h0, w_xh, w_hh, b_h):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for level04 verification") from exc

    x_t = torch.from_numpy(x)
    h_prev = torch.from_numpy(h0)
    w_xh_t = torch.from_numpy(w_xh)
    w_hh_t = torch.from_numpy(w_hh)
    b_h_t = torch.from_numpy(b_h)

    outs = []
    for t in range(x.shape[0]):
        h_prev = torch.tanh(x_t[t] @ w_xh_t + h_prev @ w_hh_t + b_h_t)
        outs.append(h_prev)
    return torch.stack(outs, dim=0).numpy()
