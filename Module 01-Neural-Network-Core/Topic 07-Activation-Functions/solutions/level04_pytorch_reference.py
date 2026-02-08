"""
Topic 07: Activation Functions - Level 04 PyTorch Reference
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


from level02_vectorized import ReLU, Sigmoid, Tanh, Softmax, softmax


def compare_activations():
    """Compare our implementations with PyTorch."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available")
        return
    
    print("=" * 50)
    print("Comparing with PyTorch")
    print("=" * 50)
    
    np.random.seed(42)
    x_np = np.random.randn(16, 32).astype(np.float32)
    x_pt = torch.tensor(x_np, requires_grad=True)
    
    tests = []
    
    # ReLU
    relu = ReLU()
    np_out = relu.forward(x_np)
    pt_out = F.relu(x_pt)
    tests.append(("ReLU forward", np.max(np.abs(np_out - pt_out.detach().numpy()))))
    
    # Sigmoid
    sigmoid = Sigmoid()
    np_out = sigmoid.forward(x_np)
    pt_out = torch.sigmoid(x_pt)
    tests.append(("Sigmoid forward", np.max(np.abs(np_out - pt_out.detach().numpy()))))
    
    # Tanh
    tanh = Tanh()
    np_out = tanh.forward(x_np)
    pt_out = torch.tanh(x_pt)
    tests.append(("Tanh forward", np.max(np.abs(np_out - pt_out.detach().numpy()))))
    
    # Softmax
    np_out = softmax(x_np)
    pt_out = F.softmax(x_pt, dim=-1)
    tests.append(("Softmax forward", np.max(np.abs(np_out - pt_out.detach().numpy()))))
    
    # Report
    all_pass = True
    for name, diff in tests:
        status = "✓" if diff < 1e-5 else "✗"
        if diff >= 1e-5:
            all_pass = False
        print(f"{name}: max diff = {diff:.2e} {status}")
    
    print(f"\nAll tests passed: {all_pass}")


def numerical_gradient_check():
    """Verify gradients numerically."""
    print("\n" + "=" * 50)
    print("Numerical Gradient Check")
    print("=" * 50)
    
    from level02_vectorized import ReLU, Sigmoid, Tanh
    
    def check_gradient(activation, name, x):
        eps = 1e-5
        
        # Analytical gradient
        _ = activation.forward(x.copy())
        grad_out = np.ones_like(x)
        analytical = activation.backward(grad_out)
        
        # Numerical gradient
        numerical = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy()
            x_plus.flat[i] += eps
            x_minus = x.copy()
            x_minus.flat[i] -= eps
            
            # Recompute forward each time
            out_plus = activation.forward(x_plus).sum()
            out_minus = activation.forward(x_minus).sum()
            numerical.flat[i] = (out_plus - out_minus) / (2 * eps)
        
        rel_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
        max_error = rel_error.max()
        status = "✓" if max_error < 1e-4 else "✗"
        print(f"{name}: max rel error = {max_error:.2e} {status}")
        return max_error < 1e-4
    
    x = np.random.randn(4, 5).astype(np.float64)
    
    results = [
        check_gradient(ReLU(), "ReLU", x),
        check_gradient(Sigmoid(), "Sigmoid", x),
        check_gradient(Tanh(), "Tanh", x),
    ]
    
    print(f"\nAll gradient checks passed: {all(results)}")


def demo():
    print("=" * 60)
    print("Activation Functions - Level 04 (PyTorch Reference)")
    print("=" * 60)
    
    if PYTORCH_AVAILABLE:
        compare_activations()
    
    numerical_gradient_check()


if __name__ == "__main__":
    demo()
