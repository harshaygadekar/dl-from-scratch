"""
Topic 09: Regularization - Level 04 PyTorch Reference
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from level02_vectorized import BatchNorm1d, Dropout


def compare_batchnorm():
    if not PYTORCH_AVAILABLE:
        return
    
    print("BatchNorm Comparison")
    print("-" * 40)
    
    np.random.seed(42)
    x = np.random.randn(32, 64).astype(np.float32)
    
    # NumPy
    np_bn = BatchNorm1d(64)
    np_out = np_bn.forward(x)
    
    # PyTorch
    pt_bn = nn.BatchNorm1d(64, affine=True)
    pt_bn.eval()
    pt_bn.train()
    with torch.no_grad():
        pt_out = pt_bn(torch.tensor(x)).numpy()
    
    diff = np.abs(np_out - pt_out).max()
    print(f"Max difference: {diff:.6f}")
    print("BatchNorm:", "PASSED ✓" if diff < 1e-4 else "FAILED ✗")


def compare_dropout():
    if not PYTORCH_AVAILABLE:
        return
    
    print("\nDropout Comparison")
    print("-" * 40)
    
    # Just verify dropout rate
    np_drop = Dropout(p=0.3)
    x = np.ones((1000, 100))
    out = np_drop.forward(x)
    zero_rate = np.mean(out == 0)
    print(f"Expected zero rate: 30%, Actual: {100*zero_rate:.1f}%")
    print("Dropout:", "PASSED ✓" if abs(zero_rate - 0.3) < 0.05 else "FAILED ✗")


def demo():
    print("=" * 60)
    print("Regularization - Level 04 (PyTorch Reference)")
    print("=" * 60)
    
    if PYTORCH_AVAILABLE:
        compare_batchnorm()
        compare_dropout()
    else:
        print("PyTorch not available")


if __name__ == "__main__":
    demo()
