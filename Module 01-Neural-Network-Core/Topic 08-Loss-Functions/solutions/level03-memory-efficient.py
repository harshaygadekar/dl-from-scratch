"""
Topic 08: Loss Functions - Level 03 Memory-Efficient

Production losses with label smoothing and focal loss.
"""

import numpy as np


class SoftmaxCrossEntropyLoss:
    """Softmax + CE with label smoothing support."""
    
    def __init__(self, label_smoothing=0.0):
        self.smoothing = label_smoothing
    
    def forward(self, logits, y_true):
        n_classes = logits.shape[-1]
        
        # Label smoothing
        if self.smoothing > 0:
            y_true = y_true * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Stable softmax
        x_max = logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(logits - x_max)
        self.probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
        self.y_true = y_true
        
        eps = 1e-15
        return -np.sum(y_true * np.log(self.probs + eps)) / len(y_true)
    
    def backward(self):
        return (self.probs - self.y_true) / len(self.y_true)


class FocalLoss:
    """Focal loss for imbalanced classification."""
    
    def __init__(self, gamma=2.0, alpha=None):
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, y_true):
        # Softmax
        x_max = logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(logits - x_max)
        self.probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
        self.y_true = y_true
        
        # p_t = prob of true class
        p_t = (self.probs * y_true).sum(axis=-1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        eps = 1e-15
        ce = -np.log(p_t + eps)
        loss = focal_weight * ce
        
        return np.mean(loss)
    
    def backward(self):
        p_t = (self.probs * self.y_true).sum(axis=-1, keepdims=True)
        focal_weight = (1 - p_t) ** self.gamma
        
        grad = focal_weight * (self.probs - self.y_true)
        grad += self.gamma * ((1 - p_t) ** (self.gamma - 1)) * p_t * self.y_true
        
        return grad / len(self.y_true)


class BCEWithLogitsLoss:
    """BCE with class weighting."""
    
    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight
    
    def forward(self, logits, y_true):
        self.logits = logits
        self.y_true = y_true
        
        # Weighted BCE
        loss = np.maximum(logits, 0) - logits * y_true + np.log(1 + np.exp(-np.abs(logits)))
        weight = y_true * self.pos_weight + (1 - y_true)
        
        return np.mean(loss * weight)
    
    def backward(self):
        sigmoid = 1 / (1 + np.exp(-self.logits))
        weight = self.y_true * self.pos_weight + (1 - self.y_true)
        return (sigmoid - self.y_true) * weight / self.logits.size


def demo():
    """Demonstrate production losses."""
    print("=" * 50)
    print("Loss Functions - Level 03 (Memory-Efficient)")
    print("=" * 50)
    
    np.random.seed(42)
    logits = np.random.randn(32, 10)
    labels = np.eye(10)[np.random.randint(0, 10, 32)]
    
    # Standard vs Label Smoothing
    ce = SoftmaxCrossEntropyLoss(label_smoothing=0.0)
    ce_smooth = SoftmaxCrossEntropyLoss(label_smoothing=0.1)
    
    print(f"CE Loss: {ce.forward(logits, labels):.4f}")
    print(f"CE + Smoothing (0.1): {ce_smooth.forward(logits, labels):.4f}")
    
    # Focal loss
    focal = FocalLoss(gamma=2.0)
    print(f"Focal Loss (γ=2): {focal.forward(logits, labels):.4f}")


if __name__ == "__main__":
    demo()
