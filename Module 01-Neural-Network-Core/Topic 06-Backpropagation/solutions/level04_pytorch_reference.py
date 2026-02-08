"""
Topic 06: Backpropagation - Level 04 PyTorch Reference

Verify our backprop against PyTorch autograd.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class NumpyLinear:
    """Our NumPy linear layer."""
    
    def __init__(self, W, b):
        self.W = W.copy()
        self.b = b.copy()
    
    def forward(self, x):
        self.input = x
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output
        self.grad_b = grad_output.sum(axis=0)
        return grad_output @ self.W.T


def compare_linear_gradients():
    """Compare linear layer gradients."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available")
        return
    
    print("=" * 50)
    print("Linear Layer Gradient Comparison")
    print("=" * 50)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create same weights
    in_f, out_f = 10, 5
    W = np.random.randn(in_f, out_f).astype(np.float32)
    b = np.random.randn(out_f).astype(np.float32)
    
    # NumPy layer
    np_layer = NumpyLinear(W, b)
    
    # PyTorch layer
    pt_layer = nn.Linear(in_f, out_f)
    with torch.no_grad():
        pt_layer.weight.copy_(torch.tensor(W.T))
        pt_layer.bias.copy_(torch.tensor(b))
    
    # Input
    x = np.random.randn(8, in_f).astype(np.float32)
    
    # NumPy forward + backward
    np_out = np_layer.forward(x)
    grad_output = np.ones_like(np_out)
    np_layer.backward(grad_output)
    
    # PyTorch forward + backward
    x_pt = torch.tensor(x, requires_grad=True)
    pt_out = pt_layer(x_pt)
    pt_out.backward(torch.ones_like(pt_out))
    
    print(f"Forward max diff: {np.max(np.abs(np_out - pt_out.detach().numpy())):.10f}")
    print(f"grad_W max diff: {np.max(np.abs(np_layer.grad_W - pt_layer.weight.grad.T.numpy())):.10f}")
    print(f"grad_b max diff: {np.max(np.abs(np_layer.grad_b - pt_layer.bias.grad.numpy())):.10f}")


def compare_mlp_gradients():
    """Compare full MLP gradients."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("\n" + "=" * 50)
    print("MLP Gradient Comparison")
    print("=" * 50)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create networks with same architecture
    from level02_vectorized import MLP as NumpyMLP, one_hot
    
    np_mlp = NumpyMLP([10, 8, 4])
    
    pt_mlp = nn.Sequential(
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 4)
    )
    
    # Copy weights
    with torch.no_grad():
        pt_mlp[0].weight.copy_(torch.tensor(np_mlp.layers[0].W.T))
        pt_mlp[0].bias.copy_(torch.tensor(np_mlp.layers[0].b))
        pt_mlp[2].weight.copy_(torch.tensor(np_mlp.layers[1].W.T))
        pt_mlp[2].bias.copy_(torch.tensor(np_mlp.layers[1].b))
    
    # Data
    x = np.random.randn(16, 10).astype(np.float32)
    y = one_hot(np.random.randint(0, 4, 16), 4).astype(np.float32)
    
    # NumPy forward + backward
    loss_np = np_mlp.forward(x, y)
    np_mlp.backward()
    
    # PyTorch forward + backward
    x_pt = torch.tensor(x)
    y_pt = torch.tensor(y)
    logits_pt = pt_mlp(x_pt)
    loss_pt = nn.functional.cross_entropy(logits_pt, y_pt.argmax(dim=1))
    loss_pt.backward()
    
    print(f"Loss - NumPy: {loss_np:.6f}, PyTorch: {loss_pt.item():.6f}")
    
    # Compare first layer gradients
    np_grad_W0 = np_mlp.layers[0].grad_W
    pt_grad_W0 = pt_mlp[0].weight.grad.T.numpy()
    print(f"Layer 0 grad_W max diff: {np.max(np.abs(np_grad_W0 - pt_grad_W0)):.6f}")


def numerical_gradient_check():
    """Verify gradients numerically."""
    print("\n" + "=" * 50)
    print("Numerical Gradient Check")
    print("=" * 50)
    
    from level02_vectorized import MLP, one_hot
    
    np.random.seed(42)
    mlp = MLP([5, 4, 3])
    
    x = np.random.randn(4, 5).astype(np.float32)
    y = one_hot(np.random.randint(0, 3, 4), 3).astype(np.float32)
    
    # Analytical gradient
    mlp.forward(x, y)
    mlp.backward()
    analytical_grad = mlp.layers[0].grad_W.copy()
    
    # Numerical gradient
    eps = 1e-5
    numerical_grad = np.zeros_like(mlp.layers[0].W)
    
    for i in range(mlp.layers[0].W.shape[0]):
        for j in range(mlp.layers[0].W.shape[1]):
            mlp.layers[0].W[i, j] += eps
            loss_plus = mlp.forward(x, y)
            mlp.layers[0].W[i, j] -= 2 * eps
            loss_minus = mlp.forward(x, y)
            mlp.layers[0].W[i, j] += eps
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
    
    rel_error = np.abs(analytical_grad - numerical_grad) / (np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-8)
    print(f"Max relative error: {rel_error.max():.10f}")
    print("Gradient check:", "PASSED ✓" if rel_error.max() < 1e-4 else "FAILED ✗")


def demo():
    print("=" * 60)
    print("Backpropagation - Level 04 (PyTorch Reference)")
    print("=" * 60)
    
    compare_linear_gradients()
    
    if PYTORCH_AVAILABLE:
        compare_mlp_gradients()
    
    numerical_gradient_check()


if __name__ == "__main__":
    demo()
