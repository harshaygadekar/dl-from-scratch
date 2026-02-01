"""
Topic 07: Activation Functions - Level 02 Vectorized

All common activations with vectorized operations.
"""

import numpy as np


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * self.mask


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, grad_output):
        return np.where(self.x > 0, grad_output, self.alpha * grad_output)


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        self.x = x
        self.output = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return self.output
    
    def backward(self, grad_output):
        return np.where(self.x > 0, grad_output, 
                       grad_output * (self.output + self.alpha))


class GELU:
    """Gaussian Error Linear Unit (used in transformers)."""
    
    def forward(self, x):
        self.x = x
        self.cdf = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        return x * self.cdf
    
    def backward(self, grad_output):
        # Approximate gradient
        x = self.x
        tanh_arg = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
        sech2 = 1 - np.tanh(tanh_arg)**2
        d_cdf = 0.5 * sech2 * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
        return grad_output * (self.cdf + x * d_cdf)


class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)


class Softmax:
    def forward(self, x, axis=-1):
        x_max = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.output = exp_x / exp_x.sum(axis=axis, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        # Full Jacobian computation (usually combined with CE)
        s = self.output
        return s * (grad_output - (grad_output * s).sum(axis=-1, keepdims=True))


class SoftmaxCrossEntropy:
    """Combined for numerical stability."""
    
    def forward(self, logits, y_true):
        x_max = logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(logits - x_max)
        self.probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
        self.y_true = y_true
        
        eps = 1e-15
        loss = -np.sum(y_true * np.log(self.probs + eps)) / len(y_true)
        return loss
    
    def backward(self):
        return (self.probs - self.y_true) / len(self.y_true)


def softmax(x, axis=-1):
    """Functional softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


# Functional versions
def relu(x): return np.maximum(0, x)
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def tanh(x): return np.tanh(x)


def demo():
    """Test all activations."""
    print("=" * 50)
    print("Activation Functions - Level 02 (Vectorized)")
    print("=" * 50)
    
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    
    activations = [
        ("ReLU", ReLU()),
        ("LeakyReLU", LeakyReLU(0.1)),
        ("ELU", ELU()),
        ("GELU", GELU()),
        ("Sigmoid", Sigmoid()),
        ("Tanh", Tanh()),
    ]
    
    print(f"\nInput: {x}")
    for name, act in activations:
        out = act.forward(x)
        grad = act.backward(np.ones_like(x))
        print(f"\n{name}:")
        print(f"  Forward:  {np.round(out, 3)}")
        print(f"  Gradient: {np.round(grad, 3)}")
    
    # Softmax + CE
    print("\nSoftmax + Cross-Entropy:")
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 3.0]])
    labels = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    
    sce = SoftmaxCrossEntropy()
    loss = sce.forward(logits, labels)
    grad = sce.backward()
    print(f"  Logits:\n{logits}")
    print(f"  Probs:\n{np.round(sce.probs, 3)}")
    print(f"  Loss: {loss:.4f}")
    print(f"  Gradient:\n{np.round(grad, 4)}")


if __name__ == "__main__":
    demo()
