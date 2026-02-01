"""
Topic 07: Activation Functions - Level 03 Memory-Efficient

Production-quality activations with in-place options and memory optimization.
"""

import numpy as np


class ReLU:
    def forward(self, x, inplace=False):
        if inplace:
            self.mask = x > 0
            x[~self.mask] = 0
            return x
        else:
            self.mask = x > 0
            return np.maximum(0, x)
    
    def backward(self, grad_output, inplace=False):
        if inplace:
            grad_output[~self.mask] = 0
            return grad_output
        return grad_output * self.mask


class Sigmoid:
    def forward(self, x):
        # Stable sigmoid
        positive = x >= 0
        result = np.empty_like(x)
        result[positive] = 1 / (1 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        result[~positive] = exp_x / (1 + exp_x)
        self.output = result
        return result
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        grad = 1 - self.output ** 2
        grad *= grad_output
        return grad


class SoftmaxCE:
    """Memory-efficient softmax + cross-entropy."""
    
    def forward(self, logits, y_true, reduce='mean'):
        # LogSumExp trick
        x_max = logits.max(axis=-1, keepdims=True)
        log_sum_exp = x_max.squeeze(-1) + np.log(np.exp(logits - x_max).sum(axis=-1))
        
        # Cross-entropy = -log(p[correct])
        correct_logits = (logits * y_true).sum(axis=-1)
        losses = log_sum_exp - correct_logits
        
        self.probs = np.exp(logits - x_max) / np.exp(logits - x_max).sum(axis=-1, keepdims=True)
        self.y_true = y_true
        
        if reduce == 'mean':
            return losses.mean()
        return losses
    
    def backward(self):
        return (self.probs - self.y_true) / len(self.y_true)


def demo():
    """Demonstrate memory-efficient activations."""
    print("=" * 50)
    print("Activation Functions - Level 03 (Memory-Efficient)")
    print("=" * 50)
    
    x = np.random.randn(1000, 100)
    
    relu = ReLU()
    out = relu.forward(x.copy(), inplace=True)
    print(f"ReLU inplace: shape={out.shape}")
    
    sigmoid = Sigmoid()
    out = sigmoid.forward(np.array([-1000, 0, 1000]))
    print(f"Stable sigmoid: {out}")
    
    # Softmax + CE
    logits = np.random.randn(32, 10)
    labels = np.eye(10)[np.random.randint(0, 10, 32)]
    
    sce = SoftmaxCE()
    loss = sce.forward(logits, labels)
    grad = sce.backward()
    print(f"Loss: {loss:.4f}")


if __name__ == "__main__":
    demo()
