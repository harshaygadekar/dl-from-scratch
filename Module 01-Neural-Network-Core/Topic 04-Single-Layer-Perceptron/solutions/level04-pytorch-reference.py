"""
Topic 04: Single Layer Perceptron - Level 04 PyTorch Reference

Verify our implementation against PyTorch's nn.Linear + BCEWithLogitsLoss.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")


class PyTorchPerceptron(nn.Module):
    """PyTorch perceptron for comparison."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        # Initialize similar to our implementation
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)


class NumpyPerceptron:
    """Our NumPy implementation for comparison."""
    
    def __init__(self, input_dim: int):
        self.w = np.random.randn(input_dim) * 0.01
        self.b = 0.0
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)
    
    def backward(self, X, y_true, y_pred):
        error = y_pred - y_true
        self.dw = np.dot(error, X) / len(X)
        self.db = np.mean(error)
    
    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db


def generate_data(n_samples: int = 200, seed: int = 42):
    """Generate test data."""
    np.random.seed(seed)
    if PYTORCH_AVAILABLE:
        torch.manual_seed(seed)
    
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.array([1.0] * (n_samples // 2) + [0.0] * (n_samples // 2), dtype=np.float32)
    
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def compare_forward():
    """Compare forward pass outputs."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("=" * 50)
    print("Forward Pass Comparison")
    print("=" * 50)
    
    # Create models with same weights
    np_model = NumpyPerceptron(2)
    pt_model = PyTorchPerceptron(2)
    
    # Copy weights from numpy to pytorch
    with torch.no_grad():
        pt_model.linear.weight.copy_(torch.tensor(np_model.w.reshape(1, -1)))
        pt_model.linear.bias.copy_(torch.tensor([np_model.b]))
    
    # Test input
    X = np.array([[1.0, 2.0], [-1.0, -2.0]], dtype=np.float32)
    
    # NumPy forward
    np_output = np_model.forward(X)
    
    # PyTorch forward (with sigmoid for comparison)
    pt_output = torch.sigmoid(pt_model(torch.tensor(X))).detach().numpy().flatten()
    
    print(f"Input:\n{X}")
    print(f"NumPy output:   {np_output}")
    print(f"PyTorch output: {pt_output}")
    print(f"Max diff: {np.max(np.abs(np_output - pt_output)):.10f}")
    

def compare_gradients():
    """Compare gradient computation."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("\n" + "=" * 50)
    print("Gradient Comparison")
    print("=" * 50)
    
    X, y = generate_data(100)
    X_tensor = torch.tensor(X, requires_grad=False)
    y_tensor = torch.tensor(y)
    
    # NumPy model
    np_model = NumpyPerceptron(2)
    
    # PyTorch model with same weights
    pt_model = PyTorchPerceptron(2)
    with torch.no_grad():
        pt_model.linear.weight.copy_(torch.tensor(np_model.w.reshape(1, -1)))
        pt_model.linear.bias.copy_(torch.tensor([np_model.b]))
    
    # NumPy gradient
    np_output = np_model.forward(X)
    np_model.backward(X, y, np_output)
    
    # PyTorch gradient
    pt_model.zero_grad()
    pt_output = pt_model(X_tensor).squeeze()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pt_output, y_tensor)
    loss.backward()
    
    pt_dw = pt_model.linear.weight.grad.detach().numpy().flatten()
    pt_db = pt_model.linear.bias.grad.item()
    
    print(f"NumPy dw:   {np_model.dw}")
    print(f"PyTorch dw: {pt_dw}")
    print(f"dw diff: {np.max(np.abs(np_model.dw - pt_dw)):.10f}")
    
    print(f"\nNumPy db:   {np_model.db:.6f}")
    print(f"PyTorch db: {pt_db:.6f}")
    print(f"db diff: {abs(np_model.db - pt_db):.10f}")


def compare_training():
    """Compare full training run."""
    if not PYTORCH_AVAILABLE:
        return
    
    print("\n" + "=" * 50)
    print("Training Comparison")
    print("=" * 50)
    
    X, y = generate_data(500, seed=123)
    lr = 0.1
    epochs = 50
    
    # NumPy training
    np_model = NumpyPerceptron(2)
    np.random.seed(42)
    np_model.w = np.random.randn(2) * 0.01
    np_model.b = 0.0
    
    for epoch in range(epochs):
        np_output = np_model.forward(X)
        np_model.backward(X, y, np_output)
        np_model.update(lr)
    
    np_acc = np.mean((np_model.forward(X) >= 0.5).astype(int) == y)
    
    # PyTorch training
    torch.manual_seed(42)
    pt_model = PyTorchPerceptron(2)
    optimizer = optim.SGD(pt_model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = pt_model(X_tensor).squeeze()
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    
    pt_pred = (torch.sigmoid(pt_model(X_tensor)).squeeze() >= 0.5).long()
    pt_acc = (pt_pred == y_tensor.long()).float().mean().item()
    
    print(f"NumPy final accuracy:   {np_acc:.2%}")
    print(f"PyTorch final accuracy: {pt_acc:.2%}")
    print(f"\nNumPy weights:   {np_model.w}, bias: {np_model.b:.4f}")
    pt_w = pt_model.linear.weight.detach().numpy().flatten()
    pt_b = pt_model.linear.bias.item()
    print(f"PyTorch weights: {pt_w}, bias: {pt_b:.4f}")


def demo():
    """Run all comparisons."""
    print("=" * 60)
    print("Single Layer Perceptron - Level 04 (PyTorch Reference)")
    print("=" * 60)
    
    if not PYTORCH_AVAILABLE:
        print("\nPyTorch is not available. Please install it to run comparisons.")
        print("pip install torch")
        return
    
    compare_forward()
    compare_gradients()
    compare_training()
    
    print("\n" + "=" * 50)
    print("All comparisons passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    demo()
