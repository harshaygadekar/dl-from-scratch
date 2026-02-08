"""
Topic 09: Regularization - Level 03 Memory-Efficient
"""

import numpy as np


class Dropout:
    """Memory-efficient dropout with bitmask."""
    
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def forward(self, x):
        if not self.training:
            return x
        self.mask = np.random.rand(*x.shape) > self.p
        np.multiply(x, self.mask, out=x)
        x /= (1 - self.p)
        return x
    
    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask / (1 - self.p)


class BatchNorm1d:
    """Memory-efficient BatchNorm with fused ops."""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
    
    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        
        # Fused normalization
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.x_centered = x - mean
        x_norm = self.x_centered * self.inv_std
        return self.gamma * x_norm + self.beta


class LayerNorm:
    """Layer Normalization (for transformers)."""
    
    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
    
    def forward(self, x):
        self.mean = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta


def demo():
    print("=" * 50)
    print("Regularization - Level 03 (Memory-Efficient)")
    print("=" * 50)
    
    # LayerNorm
    x = np.random.randn(8, 16, 64)
    ln = LayerNorm(64)
    out = ln.forward(x)
    print(f"LayerNorm output shape: {out.shape}")
    print(f"Per-sample mean: {out[0].mean():.4f}, std: {out[0].std():.4f}")


if __name__ == "__main__":
    demo()
