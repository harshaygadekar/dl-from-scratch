"""
Topic 09: Regularization - Level 02 Vectorized
"""

import numpy as np


def l2_loss(weights, lambda_=0.01):
    """Vectorized L2 loss."""
    return 0.5 * lambda_ * sum(np.sum(w**2) for w in weights)


def l2_grad(w, lambda_=0.01):
    """Vectorized L2 gradient."""
    return lambda_ * w


class Dropout:
    """Vectorized Dropout."""
    
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def forward(self, x):
        if not self.training:
            return x
        self.mask = np.random.rand(*x.shape) > self.p
        return x * self.mask / (1 - self.p)
    
    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask / (1 - self.p)


class BatchNorm1d:
    """Vectorized Batch Normalization."""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
    
    def forward(self, x):
        self.x = x
        
        if self.training:
            self.mean = x.mean(axis=0)
            self.var = x.var(axis=0)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var
        
        self.std = np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mean) / self.std
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad_output):
        N = grad_output.shape[0]
        
        self.grad_gamma = (grad_output * self.x_norm).sum(axis=0)
        self.grad_beta = grad_output.sum(axis=0)
        
        dx_norm = grad_output * self.gamma
        dvar = (dx_norm * (self.x - self.mean) * (-0.5) * (self.var + self.eps)**(-1.5)).sum(axis=0)
        dmean = (-dx_norm / self.std).sum(axis=0) + dvar * (-2/N) * (self.x - self.mean).sum(axis=0)
        
        dx = dx_norm / self.std + dvar * 2 * (self.x - self.mean) / N + dmean / N
        return dx


def demo():
    print("=" * 50)
    print("Regularization - Level 02 (Vectorized)")
    print("=" * 50)
    
    np.random.seed(42)
    
    # BatchNorm
    x = np.random.randn(32, 128) * 5 + 10
    bn = BatchNorm1d(128)
    out = bn.forward(x)
    print(f"Before BN - mean: {x.mean():.2f}, std: {x.std():.2f}")
    print(f"After BN  - mean: {out.mean():.4f}, std: {out.std():.4f}")
    
    # Dropout
    drop = Dropout(p=0.5)
    x = np.ones((100, 100))
    out = drop.forward(x)
    print(f"\nDropout: {100*np.mean(out == 0):.1f}% zeros")


if __name__ == "__main__":
    demo()
