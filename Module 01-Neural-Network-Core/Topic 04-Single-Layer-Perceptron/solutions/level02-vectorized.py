"""
Topic 04: Single Layer Perceptron - Level 02 Vectorized Implementation

Vectorized perceptron with batch training and extended features.
Uses NumPy operations instead of loops.
"""

import numpy as np
from typing import Tuple, List


class Perceptron:
    """
    Vectorized single-layer perceptron for binary classification.
    
    Supports batch training with BCE loss and sigmoid activation.
    """
    
    def __init__(self, input_dim: int):
        """Initialize perceptron."""
        self.w = np.random.randn(input_dim) * 0.01
        self.b = 0.0
        self.dw = np.zeros(input_dim)
        self.db = 0.0
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch input.
        
        Args:
            X: Input batch of shape (batch_size, input_dim)
        
        Returns:
            Predictions of shape (batch_size,)
        """
        z = np.dot(X, self.w) + self.b  # Vectorized dot product
        return self.sigmoid(z)
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute mean BCE loss over batch.
        
        Args:
            y_pred: Predictions of shape (batch_size,)
            y_true: Labels of shape (batch_size,)
        
        Returns:
            Mean loss
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(bce)
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute gradients over batch.
        
        Args:
            X: Input batch of shape (batch_size, input_dim)
            y_true: Labels of shape (batch_size,)
            y_pred: Predictions of shape (batch_size,)
        """
        batch_size = X.shape[0]
        error = y_pred - y_true  # (batch_size,)
        
        # Gradient is average over batch
        self.dw = np.dot(error, X) / batch_size  # (input_dim,)
        self.db = np.mean(error)
    
    def update(self, lr: float):
        """Update weights with gradient descent."""
        self.w -= lr * self.dw
        self.b -= lr * self.db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return (self.forward(X) >= 0.5).astype(int)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)


class PerceptronWithMomentum(Perceptron):
    """Perceptron with momentum-based optimizer."""
    
    def __init__(self, input_dim: int, momentum: float = 0.9):
        super().__init__(input_dim)
        self.momentum = momentum
        self.vw = np.zeros(input_dim)
        self.vb = 0.0
    
    def update(self, lr: float):
        """Update with momentum."""
        self.vw = self.momentum * self.vw + self.dw
        self.vb = self.momentum * self.vb + self.db
        self.w -= lr * self.vw
        self.b -= lr * self.vb


def generate_data(n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate linearly separable data."""
    np.random.seed(seed)
    
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def train_batch(X: np.ndarray, y: np.ndarray, 
                epochs: int = 100, lr: float = 0.1,
                batch_size: int = 32) -> Tuple[Perceptron, List[float]]:
    """
    Train with mini-batch gradient descent.
    
    Args:
        X: Training data
        y: Labels
        epochs: Number of epochs
        lr: Learning rate
        batch_size: Mini-batch size
    
    Returns:
        Trained model and loss history
    """
    model = Perceptron(X.shape[1])
    history = []
    n_samples = X.shape[0]
    
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_pred, y_batch)
            epoch_loss += loss
            n_batches += 1
            
            # Backward and update
            model.backward(X_batch, y_batch, y_pred)
            model.update(lr)
        
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)
        
        if epoch % 10 == 0:
            acc = model.accuracy(X, y)
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2%}")
    
    return model, history


def gradient_check(model: Perceptron, X: np.ndarray, y: np.ndarray, eps: float = 1e-5):
    """
    Verify gradients using finite differences.
    
    Args:
        model: Perceptron model
        X: Input batch
        y: Labels
        eps: Perturbation size
    """
    print("\nGradient Check:")
    print("-" * 40)
    
    # Compute analytical gradients
    y_pred = model.forward(X)
    model.backward(X, y, y_pred)
    analytical_dw = model.dw.copy()
    analytical_db = model.db
    
    # Check weight gradients
    for i in range(len(model.w)):
        # Numerical gradient
        model.w[i] += eps
        loss_plus = model.compute_loss(model.forward(X), y)
        model.w[i] -= 2 * eps
        loss_minus = model.compute_loss(model.forward(X), y)
        model.w[i] += eps  # Reset
        
        numerical = (loss_plus - loss_minus) / (2 * eps)
        
        diff = abs(numerical - analytical_dw[i])
        status = "✓" if diff < 1e-5 else "✗"
        print(f"dw[{i}]: numerical={numerical:.6f}, analytical={analytical_dw[i]:.6f}, diff={diff:.10f} {status}")
    
    # Check bias gradient
    model.b += eps
    loss_plus = model.compute_loss(model.forward(X), y)
    model.b -= 2 * eps
    loss_minus = model.compute_loss(model.forward(X), y)
    model.b += eps
    
    numerical = (loss_plus - loss_minus) / (2 * eps)
    diff = abs(numerical - analytical_db)
    status = "✓" if diff < 1e-5 else "✗"
    print(f"db: numerical={numerical:.6f}, analytical={analytical_db:.6f}, diff={diff:.10f} {status}")


def demo():
    """Demonstrate vectorized perceptron."""
    print("=" * 50)
    print("Single Layer Perceptron - Level 02 (Vectorized)")
    print("=" * 50)
    
    # Generate data
    X, y = generate_data(1000)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Gradient check first
    model = Perceptron(X.shape[1])
    gradient_check(model, X[:32], y[:32])
    
    # Train with mini-batches
    print("\nTraining with mini-batch gradient descent...")
    model, history = train_batch(X, y, epochs=100, lr=0.1, batch_size=32)
    
    # Final results
    print(f"\nFinal accuracy: {model.accuracy(X, y):.2%}")
    print(f"Learned weights: w = {model.w}")
    print(f"Learned bias: b = {model.b:.4f}")


if __name__ == "__main__":
    demo()
