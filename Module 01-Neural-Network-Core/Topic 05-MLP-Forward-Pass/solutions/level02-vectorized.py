"""
Topic 05: MLP Forward Pass - Level 02 Vectorized Implementation

Vectorized MLP with batch support and multiple activation options.
"""

import numpy as np
from typing import List, Callable, Optional


# Activation functions
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

ACTIVATIONS = {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'leaky_relu': leaky_relu
}


# Initialization functions
def xavier_init(n_in: int, n_out: int) -> np.ndarray:
    std = np.sqrt(2.0 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std

def kaiming_init(n_in: int, n_out: int) -> np.ndarray:
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_in, n_out) * std

def lecun_init(n_in: int, n_out: int) -> np.ndarray:
    std = np.sqrt(1.0 / n_in)
    return np.random.randn(n_in, n_out) * std

INITS = {
    'xavier': xavier_init,
    'kaiming': kaiming_init,
    'lecun': lecun_init
}


class Linear:
    """Vectorized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, 
                 init: str = 'kaiming'):
        self.in_features = in_features
        self.out_features = out_features
        
        init_fn = INITS.get(init, kaiming_init)
        self.W = init_fn(in_features, out_features)
        self.b = np.zeros(out_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass (vectorized).
        
        Args:
            x: Input of shape (batch, in_features) or (in_features,)
        
        Returns:
            Output of shape (batch, out_features) or (out_features,)
        """
        return np.dot(x, self.W) + self.b
    
    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"


class MLP:
    """Multi-Layer Perceptron with batch support."""
    
    def __init__(self, layer_sizes: List[int], 
                 activation: str = 'relu',
                 init: str = 'kaiming'):
        """
        Initialize MLP.
        
        Args:
            layer_sizes: Layer dimensions, e.g., [784, 256, 128, 10]
            activation: Activation function name
            init: Weight initialization scheme
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.activation = ACTIVATIONS.get(activation, relu)
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Linear(layer_sizes[i], layer_sizes[i + 1], init=init)
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            x: Input of shape (batch, input_dim) or (input_dim,)
        
        Returns:
            Output logits
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        logits = self.forward(x)
        if logits.ndim == 1:
            return np.argmax(logits)
        return np.argmax(logits, axis=1)
    
    def parameters(self):
        """Return all parameters as a list."""
        params = []
        for layer in self.layers:
            params.extend([layer.W, layer.b])
        return params
    
    def num_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.size for p in self.parameters())
    
    def __repr__(self):
        layers_str = " -> ".join(str(s) for s in self.layer_sizes)
        return f"MLP({layers_str}, activation={self.activation_name})"


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def demo():
    """Demonstrate vectorized MLP."""
    print("=" * 50)
    print("MLP Forward Pass - Level 02 (Vectorized)")
    print("=" * 50)
    
    # Create MLP
    mlp = MLP([784, 256, 128, 10], activation='relu', init='kaiming')
    print(f"\nNetwork: {mlp}")
    print(f"Total parameters: {mlp.num_parameters():,}")
    
    # Batch forward pass
    batch_size = 32
    X = np.random.randn(batch_size, 784)
    
    logits = mlp.forward(X)
    print(f"\nBatch input shape: {X.shape}")
    print(f"Batch output shape: {logits.shape}")
    
    probs = softmax(logits)
    predictions = mlp.predict(X)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Compare activations
    print("\n" + "=" * 50)
    print("Activation Comparison")
    print("=" * 50)
    
    for act_name in ['relu', 'sigmoid', 'tanh', 'leaky_relu']:
        mlp = MLP([10, 50, 50, 5], activation=act_name)
        x = np.random.randn(100, 10)
        out = mlp.forward(x)
        print(f"{act_name:12s}: output mean={out.mean():.4f}, std={out.std():.4f}")


if __name__ == "__main__":
    demo()
