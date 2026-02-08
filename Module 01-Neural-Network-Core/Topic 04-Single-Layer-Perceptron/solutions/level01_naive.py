"""
Topic 04: Single Layer Perceptron - Level 01 Naive Implementation

A simple perceptron with sigmoid activation for binary classification.
Uses explicit loops and step-by-step computation for clarity.
"""

import numpy as np


class Perceptron:
    """
    Single-layer perceptron for binary classification.
    
    Uses sigmoid activation and binary cross-entropy loss.
    Gradients are computed manually (no autograd).
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize perceptron with random weights.
        
        Args:
            input_dim: Number of input features
        """
        # Small random initialization
        self.w = np.random.randn(input_dim) * 0.01
        self.b = 0.0
        
        # Gradient storage
        self.dw = np.zeros(input_dim)
        self.db = 0.0
    
    def sigmoid(self, z: float) -> float:
        """
        Sigmoid activation function.
        
        Args:
            z: Linear combination value
        
        Returns:
            Probability in (0, 1)
        """
        # Clip for numerical stability
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass: compute prediction.
        
        Args:
            x: Input vector of shape (input_dim,)
        
        Returns:
            Predicted probability
        """
        # Linear combination (dot product + bias)
        z = 0.0
        for i in range(len(x)):
            z += self.w[i] * x[i]
        z += self.b
        
        # Sigmoid activation
        y_pred = self.sigmoid(z)
        return y_pred
    
    def compute_loss(self, y_pred: float, y_true: float) -> float:
        """
        Binary cross-entropy loss.
        
        Args:
            y_pred: Predicted probability
            y_true: True label (0 or 1)
        
        Returns:
            Loss value
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def backward(self, x: np.ndarray, y_true: float, y_pred: float):
        """
        Backward pass: compute gradients manually.
        
        The gradient of BCE + sigmoid simplifies to (y_pred - y_true).
        
        Args:
            x: Input vector
            y_true: True label
            y_pred: Predicted probability
        """
        # Error signal
        error = y_pred - y_true
        
        # Gradient for weights: dL/dw = error * x
        for i in range(len(x)):
            self.dw[i] = error * x[i]
        
        # Gradient for bias: dL/db = error
        self.db = error
    
    def update(self, lr: float):
        """
        Update weights using gradient descent.
        
        Args:
            lr: Learning rate
        """
        for i in range(len(self.w)):
            self.w[i] -= lr * self.dw[i]
        self.b -= lr * self.db
    
    def predict(self, x: np.ndarray) -> int:
        """
        Predict class label.
        
        Args:
            x: Input vector
        
        Returns:
            Predicted class (0 or 1)
        """
        y_pred = self.forward(x)
        return 1 if y_pred >= 0.5 else 0
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy on a dataset.
        
        Args:
            X: Input data of shape (n_samples, input_dim)
            y: Labels of shape (n_samples,)
        
        Returns:
            Accuracy percentage
        """
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correct += 1
        return correct / len(X)


def generate_data(n_samples: int = 200, seed: int = 42):
    """
    Generate linearly separable binary classification data.
    
    Args:
        n_samples: Total number of samples
        seed: Random seed
    
    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels of shape (n_samples,)
    """
    np.random.seed(seed)
    
    # Positive class: centered at (2, 2)
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    
    # Negative class: centered at (-2, -2)
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def train(X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.1):
    """
    Train a perceptron on the given data.
    
    Args:
        X: Training data of shape (n_samples, input_dim)
        y: Labels of shape (n_samples,)
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained perceptron and loss history
    """
    model = Perceptron(X.shape[1])
    history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for i in range(len(X)):
            # Forward pass
            y_pred = model.forward(X[i])
            
            # Compute loss
            loss = model.compute_loss(y_pred, y[i])
            total_loss += loss
            
            # Backward pass
            model.backward(X[i], y[i], y_pred)
            
            # Update weights
            model.update(lr)
        
        avg_loss = total_loss / len(X)
        history.append(avg_loss)
        
        if epoch % 10 == 0:
            acc = model.accuracy(X, y)
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Accuracy = {acc:.2%}")
    
    return model, history


def demo():
    """Demonstrate perceptron training."""
    print("=" * 50)
    print("Single Layer Perceptron - Level 01 (Naive)")
    print("=" * 50)
    
    # Generate data
    X, y = generate_data(200)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train
    print("\nTraining...")
    model, history = train(X, y, epochs=100, lr=0.1)
    
    # Final results
    print(f"\nFinal accuracy: {model.accuracy(X, y):.2%}")
    print(f"Learned weights: w = {model.w}")
    print(f"Learned bias: b = {model.b:.4f}")


if __name__ == "__main__":
    demo()
