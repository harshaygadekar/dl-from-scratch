"""
Topic 08: Loss Functions - Level 01 Naive Implementation

All loss functions with explicit loops.
"""

import numpy as np


class MSELoss:
    """Mean Squared Error loss."""
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        
        total = 0.0
        for i in range(y_pred.size):
            diff = y_pred.flat[i] - y_true.flat[i]
            total += diff * diff
        
        return total / y_pred.size
    
    def backward(self):
        grad = np.zeros_like(self.y_pred)
        n = self.y_pred.size
        
        for i in range(n):
            grad.flat[i] = 2 * (self.y_pred.flat[i] - self.y_true.flat[i]) / n
        
        return grad


class MAELoss:
    """Mean Absolute Error loss."""
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        
        total = 0.0
        for i in range(y_pred.size):
            total += abs(y_pred.flat[i] - y_true.flat[i])
        
        return total / y_pred.size
    
    def backward(self):
        grad = np.zeros_like(self.y_pred)
        n = self.y_pred.size
        
        for i in range(n):
            diff = self.y_pred.flat[i] - self.y_true.flat[i]
            grad.flat[i] = (1 if diff > 0 else (-1 if diff < 0 else 0)) / n
        
        return grad


class CrossEntropyLoss:
    """Cross-Entropy loss (expects softmax probabilities)."""
    
    def forward(self, probs, y_true):
        self.probs = probs
        self.y_true = y_true
        
        batch_size = probs.shape[0]
        num_classes = probs.shape[1]
        
        total_loss = 0.0
        for b in range(batch_size):
            for c in range(num_classes):
                if y_true[b, c] > 0:
                    p = max(probs[b, c], 1e-15)
                    total_loss -= y_true[b, c] * np.log(p)
        
        return total_loss / batch_size
    
    def backward(self):
        batch_size = self.probs.shape[0]
        num_classes = self.probs.shape[1]
        
        grad = np.zeros_like(self.probs)
        for b in range(batch_size):
            for c in range(num_classes):
                if self.y_true[b, c] > 0:
                    p = max(self.probs[b, c], 1e-15)
                    grad[b, c] = -self.y_true[b, c] / p / batch_size
        
        return grad


class BCELoss:
    """Binary Cross-Entropy loss."""
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        
        total = 0.0
        for i in range(y_pred.size):
            p = max(min(y_pred.flat[i], 1 - 1e-15), 1e-15)
            y = y_true.flat[i]
            total -= y * np.log(p) + (1 - y) * np.log(1 - p)
        
        return total / y_pred.size
    
    def backward(self):
        grad = np.zeros_like(self.y_pred)
        n = self.y_pred.size
        
        for i in range(n):
            p = max(min(self.y_pred.flat[i], 1 - 1e-15), 1e-15)
            y = self.y_true.flat[i]
            grad.flat[i] = (p - y) / (p * (1 - p)) / n
        
        return grad


def demo():
    """Test all losses."""
    print("=" * 50)
    print("Loss Functions - Level 01 (Naive)")
    print("=" * 50)
    
    # Regression losses
    y_pred = np.array([2.5, 0.0, 2.0])
    y_true = np.array([3.0, -0.5, 2.0])
    
    mse = MSELoss()
    mae = MAELoss()
    
    print(f"\nRegression:")
    print(f"  y_pred: {y_pred}")
    print(f"  y_true: {y_true}")
    print(f"  MSE: {mse.forward(y_pred, y_true):.4f}")
    print(f"  MAE: {mae.forward(y_pred, y_true):.4f}")
    
    # Classification losses
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    labels = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    
    ce = CrossEntropyLoss()
    print(f"\nClassification:")
    print(f"  CE Loss: {ce.forward(probs, labels):.4f}")
    
    # Binary
    p = np.array([0.9, 0.1, 0.8])
    y = np.array([1.0, 0.0, 1.0])
    
    bce = BCELoss()
    print(f"\nBinary:")
    print(f"  BCE Loss: {bce.forward(p, y):.4f}")


if __name__ == "__main__":
    demo()
