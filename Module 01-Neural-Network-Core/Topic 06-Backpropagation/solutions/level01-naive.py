"""
Topic 06: Backpropagation - Level 01 Naive Implementation

Simple backprop with explicit loops for understanding.
"""

import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


class LinearNaive:
    """Linear layer with explicit loop-based backward pass."""
    
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None
        self.input = None
    
    def forward(self, x):
        """Forward pass: y = Wx + b"""
        self.input = x
        output = np.zeros(self.W.shape[1])
        for j in range(self.W.shape[1]):
            total = 0.0
            for i in range(self.W.shape[0]):
                total += x[i] * self.W[i, j]
            output[j] = total + self.b[j]
        return output
    
    def backward(self, grad_output):
        """
        Backward pass with explicit loops.
        
        Args:
            grad_output: ∂L/∂y, gradient from next layer
        
        Returns:
            grad_input: ∂L/∂x, gradient to pass to previous layer
        """
        x = self.input
        
        # Compute weight gradient: ∂L/∂W
        self.grad_W = np.zeros_like(self.W)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                # ∂L/∂W_ij = x_i × ∂L/∂y_j
                self.grad_W[i, j] = x[i] * grad_output[j]
        
        # Compute bias gradient: ∂L/∂b
        self.grad_b = np.zeros_like(self.b)
        for j in range(len(self.b)):
            self.grad_b[j] = grad_output[j]
        
        # Compute input gradient: ∂L/∂x
        grad_input = np.zeros_like(x)
        for i in range(len(x)):
            total = 0.0
            for j in range(len(grad_output)):
                # ∂L/∂x_i = Σⱼ W_ij × ∂L/∂y_j
                total += self.W[i, j] * grad_output[j]
            grad_input[i] = total
        
        return grad_input


class SigmoidNaive:
    """Sigmoid activation with explicit backward."""
    
    def forward(self, x):
        self.input = x
        self.output = sigmoid(x)
        return self.output
    
    def backward(self, grad_output):
        """∂L/∂x = ∂L/∂y × σ(x)(1 - σ(x))"""
        grad_input = np.zeros_like(self.input)
        for i in range(len(self.input)):
            local_grad = self.output[i] * (1 - self.output[i])
            grad_input[i] = grad_output[i] * local_grad
        return grad_input


class BCELossNaive:
    """Binary Cross-Entropy loss."""
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self):
        """∂L/∂ŷ = (ŷ - y) / (ŷ(1-ŷ))"""
        eps = 1e-15
        y_pred = np.clip(self.y_pred, eps, 1 - eps)
        grad = (y_pred - self.y_true) / (y_pred * (1 - y_pred) + eps)
        return grad / len(self.y_true)


class SimpleNetwork:
    """Simple 2-layer network for demonstration."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = LinearNaive(input_dim, hidden_dim)
        self.sigmoid1 = SigmoidNaive()
        self.linear2 = LinearNaive(hidden_dim, output_dim)
        self.sigmoid2 = SigmoidNaive()
        self.loss_fn = BCELossNaive()
    
    def forward(self, x, y):
        """Forward pass."""
        z1 = self.linear1.forward(x)
        a1 = self.sigmoid1.forward(z1)
        z2 = self.linear2.forward(a1)
        a2 = self.sigmoid2.forward(z2)
        loss = self.loss_fn.forward(a2, y)
        return loss
    
    def backward(self):
        """Backward pass through entire network."""
        # Loss gradient
        grad = self.loss_fn.backward()
        
        # Backward through layers (reverse order)
        grad = self.sigmoid2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.sigmoid1.backward(grad)
        grad = self.linear1.backward(grad)
        
        return grad
    
    def update(self, lr):
        """Update weights using gradients."""
        self.linear1.W -= lr * self.linear1.grad_W
        self.linear1.b -= lr * self.linear1.grad_b
        self.linear2.W -= lr * self.linear2.grad_W
        self.linear2.b -= lr * self.linear2.grad_b


def demo():
    """Demonstrate naive backpropagation."""
    print("=" * 50)
    print("Backpropagation - Level 01 (Naive)")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Create network
    net = SimpleNetwork(2, 4, 1)
    
    # Simple XOR-like data (single samples for naive impl)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    
    print("\nTraining on XOR problem...")
    for epoch in range(1000):
        total_loss = 0
        for i in range(len(X)):
            loss = net.forward(X[i], y[i])
            net.backward()
            net.update(lr=1.0)
            total_loss += loss
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(X):.4f}")
    
    # Test predictions
    print("\nFinal predictions:")
    for i in range(len(X)):
        net.forward(X[i], y[i])
        pred = net.sigmoid2.output
        print(f"  {X[i]} -> {pred[0]:.3f} (target: {y[i][0]})")


if __name__ == "__main__":
    demo()
