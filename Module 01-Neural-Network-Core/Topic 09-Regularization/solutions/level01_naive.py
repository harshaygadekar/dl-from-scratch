"""
Topic 09: Regularization - Level 01 Naive Implementation
"""

import numpy as np


def l2_regularization_loss(weights, lambda_=0.01):
    """Compute L2 regularization loss."""
    total = 0.0
    for w in weights:
        for i in range(w.size):
            total += w.flat[i] ** 2
    return 0.5 * lambda_ * total


def l2_regularization_grad(w, lambda_=0.01):
    """Compute L2 regularization gradient."""
    grad = np.zeros_like(w)
    for i in range(w.size):
        grad.flat[i] = lambda_ * w.flat[i]
    return grad


class Dropout:
    """Dropout with explicit loops."""
    
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def forward(self, x):
        if not self.training:
            return x.copy()
        
        self.mask = np.zeros_like(x, dtype=bool)
        for i in range(x.size):
            self.mask.flat[i] = np.random.rand() > self.p
        
        output = np.zeros_like(x)
        for i in range(x.size):
            if self.mask.flat[i]:
                output.flat[i] = x.flat[i] / (1 - self.p)
        
        return output
    
    def backward(self, grad_output):
        if not self.training:
            return grad_output.copy()
        
        grad_input = np.zeros_like(grad_output)
        for i in range(grad_output.size):
            if self.mask.flat[i]:
                grad_input.flat[i] = grad_output.flat[i] / (1 - self.p)
        
        return grad_input


class BatchNorm1d:
    """Batch Normalization with explicit loops."""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
    
    def forward(self, x):
        batch_size, features = x.shape
        self.x = x
        
        if self.training:
            # Compute batch stats
            self.mean = np.zeros(features)
            for j in range(features):
                total = 0.0
                for i in range(batch_size):
                    total += x[i, j]
                self.mean[j] = total / batch_size
            
            self.var = np.zeros(features)
            for j in range(features):
                total = 0.0
                for i in range(batch_size):
                    total += (x[i, j] - self.mean[j]) ** 2
                self.var[j] = total / batch_size
            
            # Update running stats
            for j in range(features):
                self.running_mean[j] = (1-self.momentum) * self.running_mean[j] + self.momentum * self.mean[j]
                self.running_var[j] = (1-self.momentum) * self.running_var[j] + self.momentum * self.var[j]
        else:
            self.mean = self.running_mean
            self.var = self.running_var
        
        # Normalize
        self.x_norm = np.zeros_like(x)
        for i in range(batch_size):
            for j in range(features):
                self.x_norm[i, j] = (x[i, j] - self.mean[j]) / np.sqrt(self.var[j] + self.eps)
        
        # Scale and shift
        output = np.zeros_like(x)
        for i in range(batch_size):
            for j in range(features):
                output[i, j] = self.gamma[j] * self.x_norm[i, j] + self.beta[j]
        
        return output


def demo():
    print("=" * 50)
    print("Regularization - Level 01 (Naive)")
    print("=" * 50)
    
    # L2
    weights = [np.random.randn(3, 4)]
    l2_loss = l2_regularization_loss(weights, lambda_=0.01)
    print(f"\nL2 Loss: {l2_loss:.4f}")
    
    # Dropout
    x = np.random.randn(4, 8)
    dropout = Dropout(p=0.5)
    out = dropout.forward(x)
    print(f"\nDropout: {np.sum(out == 0)} zeros out of {out.size}")
    
    # BatchNorm
    x = np.random.randn(16, 64)
    bn = BatchNorm1d(64)
    out = bn.forward(x)
    print(f"\nBatchNorm output mean: {out.mean():.4f}, std: {out.std():.4f}")


if __name__ == "__main__":
    demo()
