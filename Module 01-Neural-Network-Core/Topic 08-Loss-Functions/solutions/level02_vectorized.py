"""
Topic 08: Loss Functions - Level 02 Vectorized

All common losses with vectorized operations.
"""

import numpy as np


class MSELoss:
    """Mean Squared Error."""
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.diff = y_pred - y_true
        return np.mean(self.diff ** 2)
    
    def backward(self):
        return 2 * self.diff / self.y_pred.size


class MAELoss:
    """Mean Absolute Error."""
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.diff = y_pred - y_true
        return np.mean(np.abs(self.diff))
    
    def backward(self):
        return np.sign(self.diff) / self.y_pred.size


class HuberLoss:
    """Huber loss (smooth L1)."""
    
    def __init__(self, delta=1.0):
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        self.abs_diff = np.abs(self.diff)
        
        quadratic = 0.5 * self.diff ** 2
        linear = self.delta * self.abs_diff - 0.5 * self.delta ** 2
        
        return np.mean(np.where(self.abs_diff <= self.delta, quadratic, linear))
    
    def backward(self):
        n = self.diff.size
        grad = np.where(
            self.abs_diff <= self.delta,
            self.diff,
            self.delta * np.sign(self.diff)
        )
        return grad / n


class CrossEntropyLoss:
    """Cross-Entropy (expects softmax probabilities)."""
    
    def forward(self, probs, y_true):
        self.probs = probs
        self.y_true = y_true
        
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        return -np.sum(y_true * np.log(probs)) / len(y_true)
    
    def backward(self):
        eps = 1e-15
        return -self.y_true / (self.probs + eps) / len(self.y_true)


class SoftmaxCrossEntropyLoss:
    """Softmax + Cross-Entropy combined (from logits)."""
    
    def forward(self, logits, y_true):
        # Stable softmax
        x_max = logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(logits - x_max)
        self.probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
        self.y_true = y_true
        
        eps = 1e-15
        loss = -np.sum(y_true * np.log(self.probs + eps)) / len(y_true)
        return loss
    
    def backward(self):
        return (self.probs - self.y_true) / len(self.y_true)


class BCELoss:
    """Binary Cross-Entropy (expects probabilities)."""
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)
    
    def backward(self):
        eps = 1e-15
        y_pred = np.clip(self.y_pred, eps, 1 - eps)
        return (y_pred - self.y_true) / (y_pred * (1 - y_pred)) / self.y_pred.size


class BCEWithLogitsLoss:
    """Binary Cross-Entropy with logits (stable)."""
    
    def forward(self, logits, y_true):
        self.logits = logits
        self.y_true = y_true
        
        # Stable formula
        loss = np.maximum(logits, 0) - logits * y_true + np.log(1 + np.exp(-np.abs(logits)))
        return np.mean(loss)
    
    def backward(self):
        sigmoid = 1 / (1 + np.exp(-self.logits))
        return (sigmoid - self.y_true) / self.logits.size


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def demo():
    """Test all losses."""
    print("=" * 50)
    print("Loss Functions - Level 02 (Vectorized)")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Regression
    y_pred = np.random.randn(100)
    y_true = y_pred + 0.5 * np.random.randn(100)
    
    mse = MSELoss()
    mae = MAELoss()
    huber = HuberLoss(delta=1.0)
    
    print(f"\nRegression losses (n=100):")
    print(f"  MSE: {mse.forward(y_pred, y_true):.4f}")
    print(f"  MAE: {mae.forward(y_pred, y_true):.4f}")
    print(f"  Huber: {huber.forward(y_pred, y_true):.4f}")
    
    # Classification
    logits = np.random.randn(32, 10)
    labels = np.eye(10)[np.random.randint(0, 10, 32)]
    
    sce = SoftmaxCrossEntropyLoss()
    print(f"\nClassification (batch=32, classes=10):")
    print(f"  Softmax-CE: {sce.forward(logits, labels):.4f}")
    
    # Binary
    logits_bin = np.random.randn(100)
    labels_bin = np.random.randint(0, 2, 100).astype(float)
    
    bce = BCEWithLogitsLoss()
    print(f"\nBinary (n=100):")
    print(f"  BCE (logits): {bce.forward(logits_bin, labels_bin):.4f}")


if __name__ == "__main__":
    demo()
