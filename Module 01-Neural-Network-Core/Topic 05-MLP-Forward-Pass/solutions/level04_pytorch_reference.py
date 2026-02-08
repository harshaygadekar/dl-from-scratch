"""
Topic 05: MLP Forward Pass - Level 04 PyTorch Reference

Verify our MLP against PyTorch nn.Linear and Sequential.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class NumpyMLP:
    """Our NumPy MLP for comparison."""
    
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i+1])
            self.layers.append((W, b))
    
    def forward(self, x):
        h = x
        for i, (W, b) in enumerate(self.layers):
            h = np.dot(h, W) + b
            if i < len(self.layers) - 1:
                h = np.maximum(0, h)  # ReLU
        return h


def create_pytorch_mlp(sizes):
    """Create equivalent PyTorch model."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def compare_forward():
    """Compare forward pass outputs."""
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available")
        return
    
    print("=" * 50)
    print("Forward Pass Comparison")
    print("=" * 50)
    
    sizes = [10, 8, 6, 4]
    
    np_mlp = NumpyMLP(sizes)
    pt_mlp = create_pytorch_mlp(sizes)
    
    # Copy weights from numpy to pytorch
    with torch.no_grad():
        j = 0
        for i, (name, param) in enumerate(pt_mlp.named_parameters()):
            if 'weight' in name:
                param.copy_(torch.tensor(np_mlp.layers[j][0].T, dtype=torch.float32))
            elif 'bias' in name:
                param.copy_(torch.tensor(np_mlp.layers[j][1], dtype=torch.float32))
                j += 1
    
    # Test
    x = np.random.randn(5, 10).astype(np.float32)
    np_out = np_mlp.forward(x)
    pt_out = pt_mlp(torch.tensor(x)).detach().numpy()
    
    print(f"Input shape: {x.shape}")
    print(f"NumPy output shape: {np_out.shape}")
    print(f"PyTorch output shape: {pt_out.shape}")
    print(f"Max difference: {np.max(np.abs(np_out - pt_out)):.10f}")
    
    if np.allclose(np_out, pt_out, rtol=1e-5):
        print("✓ Outputs match!")
    else:
        print("✗ Outputs differ")


def compare_init_statistics():
    """Compare initialization statistics."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("\n" + "=" * 50)
    print("Initialization Statistics Comparison")
    print("=" * 50)
    
    n_in, n_out = 512, 256
    
    # Xavier
    np.random.seed(42)
    np_xavier = np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
    
    torch.manual_seed(42)
    pt_linear = nn.Linear(n_in, n_out)
    nn.init.xavier_normal_(pt_linear.weight)
    
    print(f"Xavier - NumPy std: {np_xavier.std():.4f}")
    print(f"Xavier - PyTorch std: {pt_linear.weight.std().item():.4f}")
    
    # Kaiming
    np.random.seed(42)
    np_kaiming = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    
    torch.manual_seed(42)
    nn.init.kaiming_normal_(pt_linear.weight, mode='fan_in', nonlinearity='relu')
    
    print(f"Kaiming - NumPy std: {np_kaiming.std():.4f}")
    print(f"Kaiming - PyTorch std: {pt_linear.weight.std().item():.4f}")


def demo():
    """Run all comparisons."""
    print("=" * 60)
    print("MLP Forward Pass - Level 04 (PyTorch Reference)")
    print("=" * 60)
    
    if not PYTORCH_AVAILABLE:
        print("\nPyTorch not available. Install with: pip install torch")
        return
    
    compare_forward()
    compare_init_statistics()
    
    print("\n" + "=" * 50)
    print("All comparisons complete!")


if __name__ == "__main__":
    demo()
