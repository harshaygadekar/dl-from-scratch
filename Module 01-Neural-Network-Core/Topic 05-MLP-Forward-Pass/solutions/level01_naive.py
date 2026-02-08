"""
Topic 05: MLP Forward Pass - Level 01 Naive Implementation

Simple MLP with explicit loops for understanding.
"""

import numpy as np
from typing import List


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def xavier_init(n_in: int, n_out: int) -> np.ndarray:
    """Xavier initialization."""
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std


class Linear:
    """Single linear layer."""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.W = xavier_init(in_features, out_features)
        self.b = np.zeros(out_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass using loops (naive).
        
        Args:
            x: Input of shape (in_features,)
        
        Returns:
            Output of shape (out_features,)
        """
        output = np.zeros(self.out_features)
        for j in range(self.out_features):
            total = 0.0
            for i in range(self.in_features):
                total += x[i] * self.W[i, j]
            output[j] = total + self.b[j]
        return output


class MLP:
    """Multi-Layer Perceptron (naive implementation)."""
    
    def __init__(self, layer_sizes: List[int]):
        """
        Initialize MLP.
        
        Args:
            layer_sizes: List of layer dimensions, e.g., [784, 256, 128, 10]
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            x: Input of shape (input_dim,)
        
        Returns:
            Output logits of shape (output_dim,)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            # Apply ReLU after all layers except the last
            if i < len(self.layers) - 1:
                h = relu(h)
        return h
    
    def predict(self, x: np.ndarray) -> int:
        """Predict class label."""
        logits = self.forward(x)
        return np.argmax(logits)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)


def generate_data(n_samples: int = 200, n_features: int = 10, 
                  n_classes: int = 3, seed: int = 42):
    """Generate synthetic classification data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def demo():
    """Demonstrate MLP forward pass."""
    print("=" * 50)
    print("MLP Forward Pass - Level 01 (Naive)")
    print("=" * 50)
    
    # Create MLP: 10 inputs -> 8 hidden -> 6 hidden -> 3 outputs
    mlp = MLP([10, 8, 6, 3])
    print(f"\nNetwork architecture: 10 -> 8 -> 6 -> 3")
    print(f"Number of layers: {len(mlp.layers)}")
    
    # Generate sample input
    x = np.random.randn(10)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    logits = mlp.forward(x)
    print(f"Output logits: {logits}")
    print(f"Output shape: {logits.shape}")
    
    # Convert to probabilities
    probs = softmax(logits)
    print(f"Probabilities: {probs}")
    
    # Prediction
    pred = mlp.predict(x)
    print(f"Predicted class: {pred}")
    
    # Show layer info
    print("\nLayer details:")
    for i, layer in enumerate(mlp.layers):
        print(f"  Layer {i}: {layer.in_features} -> {layer.out_features}")
        print(f"    Weight shape: {layer.W.shape}")
        print(f"    Bias shape: {layer.b.shape}")


if __name__ == "__main__":
    demo()
