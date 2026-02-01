"""
Topic 05: MLP Forward Pass - Level 03 Memory-Efficient

Production-quality MLP with caching, lazy evaluation, and memory optimization.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple


def relu(x): return np.maximum(0, x)
def kaiming_init(n_in, n_out): return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)


class Linear:
    """Memory-efficient linear layer with activation caching."""
    
    def __init__(self, in_features: int, out_features: int):
        self.W = kaiming_init(in_features, out_features)
        self.b = np.zeros(out_features)
        self.cache: Optional[Dict] = None  # For backprop
    
    def forward(self, x: np.ndarray, cache: bool = True) -> np.ndarray:
        out = np.dot(x, self.W) + self.b
        if cache:
            self.cache = {'input': x}
        return out
    
    def clear_cache(self):
        self.cache = None


class MLP:
    """Memory-efficient MLP with gradient checkpointing support."""
    
    def __init__(self, layer_sizes: List[int]):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
        self._activations: Optional[List] = None
    
    def forward(self, x: np.ndarray, cache: bool = True) -> np.ndarray:
        """Forward with optional activation caching."""
        h = x
        if cache:
            self._activations = [x]
        
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, cache=cache)
            if i < len(self.layers) - 1:
                h = relu(h)
            if cache:
                self._activations.append(h)
        return h
    
    def forward_checkpointed(self, x: np.ndarray, checkpoint_every: int = 2) -> np.ndarray:
        """Forward with gradient checkpointing to save memory."""
        h = x
        checkpoints = [x]
        
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, cache=False)
            if i < len(self.layers) - 1:
                h = relu(h)
            if (i + 1) % checkpoint_every == 0:
                checkpoints.append(h.copy())
        
        self._activations = checkpoints
        return h
    
    def clear_cache(self):
        """Free memory."""
        for layer in self.layers:
            layer.clear_cache()
        self._activations = None
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x, cache=False)
        return np.argmax(logits, axis=-1) if logits.ndim > 1 else np.argmax(logits)
    
    def num_parameters(self) -> int:
        return sum(l.W.size + l.b.size for l in self.layers)


def demo():
    """Demonstrate memory-efficient MLP."""
    print("=" * 50)
    print("MLP Forward Pass - Level 03 (Memory-Efficient)")
    print("=" * 50)
    
    mlp = MLP([784, 512, 256, 128, 10])
    print(f"Parameters: {mlp.num_parameters():,}")
    
    X = np.random.randn(64, 784)
    
    # Normal forward
    out = mlp.forward(X, cache=True)
    print(f"\nWith caching: {len(mlp._activations)} tensors cached")
    
    # Checkpointed forward
    out = mlp.forward_checkpointed(X, checkpoint_every=2)
    print(f"With checkpointing: {len(mlp._activations)} tensors cached")
    
    mlp.clear_cache()
    print("Cache cleared.")


if __name__ == "__main__":
    demo()
