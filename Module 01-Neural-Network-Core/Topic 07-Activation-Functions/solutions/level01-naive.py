"""
Topic 07: Activation Functions - Level 01 Naive Implementation

All activations with explicit loops.
"""

import numpy as np


class ReLU:
    """ReLU activation."""
    
    def forward(self, x):
        self.input = x
        output = np.zeros_like(x)
        for i in range(x.size):
            output.flat[i] = max(0, x.flat[i])
        return output
    
    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        for i in range(self.input.size):
            if self.input.flat[i] > 0:
                grad_input.flat[i] = grad_output.flat[i]
        return grad_input


class LeakyReLU:
    """LeakyReLU with α for negative slope."""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        self.input = x
        output = np.zeros_like(x)
        for i in range(x.size):
            if x.flat[i] > 0:
                output.flat[i] = x.flat[i]
            else:
                output.flat[i] = self.alpha * x.flat[i]
        return output
    
    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        for i in range(self.input.size):
            if self.input.flat[i] > 0:
                grad_input.flat[i] = grad_output.flat[i]
            else:
                grad_input.flat[i] = self.alpha * grad_output.flat[i]
        return grad_input


class Sigmoid:
    """Sigmoid activation."""
    
    def forward(self, x):
        self.output = np.zeros_like(x)
        for i in range(x.size):
            val = max(-500, min(500, x.flat[i]))
            self.output.flat[i] = 1 / (1 + np.exp(-val))
        return self.output
    
    def backward(self, grad_output):
        grad_input = np.zeros_like(self.output)
        for i in range(self.output.size):
            y = self.output.flat[i]
            grad_input.flat[i] = grad_output.flat[i] * y * (1 - y)
        return grad_input


class Tanh:
    """Tanh activation."""
    
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        grad_input = np.zeros_like(self.output)
        for i in range(self.output.size):
            y = self.output.flat[i]
            grad_input.flat[i] = grad_output.flat[i] * (1 - y * y)
        return grad_input


class Softmax:
    """Softmax activation."""
    
    def forward(self, x):
        # Handle batched input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self.output = np.zeros_like(x)
        for b in range(x.shape[0]):
            x_max = np.max(x[b])
            exp_sum = 0
            exp_vals = np.zeros(x.shape[1])
            for j in range(x.shape[1]):
                exp_vals[j] = np.exp(x[b, j] - x_max)
                exp_sum += exp_vals[j]
            for j in range(x.shape[1]):
                self.output[b, j] = exp_vals[j] / exp_sum
        
        return self.output


def demo():
    """Test all activations."""
    print("=" * 50)
    print("Activation Functions - Level 01 (Naive)")
    print("=" * 50)
    
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    activations = [
        ("ReLU", ReLU()),
        ("LeakyReLU", LeakyReLU(0.1)),
        ("Sigmoid", Sigmoid()),
        ("Tanh", Tanh()),
    ]
    
    print(f"\nInput: {x}")
    for name, act in activations:
        out = act.forward(x)
        grad_out = np.ones_like(x)
        grad_in = act.backward(grad_out)
        print(f"\n{name}:")
        print(f"  Forward:  {out}")
        print(f"  Backward: {grad_in}")
    
    # Test softmax
    logits = np.array([[2.0, 1.0, 0.1]])
    softmax = Softmax()
    probs = softmax.forward(logits)
    print(f"\nSoftmax:")
    print(f"  Input logits: {logits[0]}")
    print(f"  Output probs: {probs[0]} (sum={probs.sum():.4f})")


if __name__ == "__main__":
    demo()
