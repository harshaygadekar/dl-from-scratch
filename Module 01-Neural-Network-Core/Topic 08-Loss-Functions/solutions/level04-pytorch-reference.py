"""
Topic 08: Loss Functions - Level 04 PyTorch Reference
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from level02_vectorized import MSELoss, SoftmaxCrossEntropyLoss, BCEWithLogitsLoss


def compare_mse():
    if not PYTORCH_AVAILABLE:
        return
    
    print("MSE Loss Comparison")
    print("-" * 40)
    
    np.random.seed(42)
    y_pred = np.random.randn(10, 5).astype(np.float32)
    y_true = np.random.randn(10, 5).astype(np.float32)
    
    np_mse = MSELoss()
    np_loss = np_mse.forward(y_pred, y_true)
    
    pt_mse = nn.MSELoss()
    pt_loss = pt_mse(torch.tensor(y_pred), torch.tensor(y_true)).item()
    
    print(f"NumPy MSE:   {np_loss:.6f}")
    print(f"PyTorch MSE: {pt_loss:.6f}")
    print(f"Difference:  {abs(np_loss - pt_loss):.10f}")


def compare_ce():
    if not PYTORCH_AVAILABLE:
        return
    
    print("\nCross-Entropy Loss Comparison")
    print("-" * 40)
    
    np.random.seed(42)
    logits = np.random.randn(16, 10).astype(np.float32)
    labels = np.random.randint(0, 10, 16)
    labels_onehot = np.eye(10)[labels].astype(np.float32)
    
    np_ce = SoftmaxCrossEntropyLoss()
    np_loss = np_ce.forward(logits, labels_onehot)
    
    pt_ce = nn.CrossEntropyLoss()
    pt_loss = pt_ce(torch.tensor(logits), torch.tensor(labels)).item()
    
    print(f"NumPy CE:   {np_loss:.6f}")
    print(f"PyTorch CE: {pt_loss:.6f}")
    print(f"Difference: {abs(np_loss - pt_loss):.6f}")


def compare_bce():
    if not PYTORCH_AVAILABLE:
        return
    
    print("\nBCE Loss Comparison")
    print("-" * 40)
    
    np.random.seed(42)
    logits = np.random.randn(100).astype(np.float32)
    labels = np.random.randint(0, 2, 100).astype(np.float32)
    
    np_bce = BCEWithLogitsLoss()
    np_loss = np_bce.forward(logits, labels)
    
    pt_bce = nn.BCEWithLogitsLoss()
    pt_loss = pt_bce(torch.tensor(logits), torch.tensor(labels)).item()
    
    print(f"NumPy BCE:   {np_loss:.6f}")
    print(f"PyTorch BCE: {pt_loss:.6f}")
    print(f"Difference:  {abs(np_loss - pt_loss):.10f}")


def numerical_gradient_check():
    print("\nNumerical Gradient Check")
    print("-" * 40)
    
    from level02_vectorized import MSELoss, SoftmaxCrossEntropyLoss
    
    # MSE gradient check
    y_pred = np.random.randn(5).astype(np.float64)
    y_true = np.random.randn(5).astype(np.float64)
    
    mse = MSELoss()
    mse.forward(y_pred, y_true)
    analytical = mse.backward()
    
    eps = 1e-5
    numerical = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_plus = y_pred.copy()
        y_plus[i] += eps
        y_minus = y_pred.copy()
        y_minus[i] -= eps
        
        mse_plus = MSELoss()
        mse_minus = MSELoss()
        numerical[i] = (mse_plus.forward(y_plus, y_true) - mse_minus.forward(y_minus, y_true)) / (2 * eps)
    
    rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    print(f"MSE max rel error: {rel_error.max():.10f}")
    print("MSE gradient check:", "PASSED ✓" if rel_error.max() < 1e-4 else "FAILED ✗")


def demo():
    print("=" * 60)
    print("Loss Functions - Level 04 (PyTorch Reference)")
    print("=" * 60)
    
    if PYTORCH_AVAILABLE:
        compare_mse()
        compare_ce()
        compare_bce()
    else:
        print("PyTorch not available")
    
    numerical_gradient_check()


if __name__ == "__main__":
    demo()
