"""
Topic 06: Backpropagation - Level 03 Memory-Efficient

Production-quality backprop with gradient checkpointing and memory optimization.
"""

import numpy as np
from typing import List, Optional


def relu(x): return np.maximum(0, x)
def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class Linear:
    """Linear layer with gradient clipping and accumulation."""
    
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self._grad_count = 0
    
    def forward(self, x, cache=True):
        if cache:
            self.input = x
        return x @ self.W + self.b
    
    def backward(self, grad_output, accumulate=False):
        if accumulate:
            self.grad_W += self.input.T @ grad_output
            self.grad_b += grad_output.sum(axis=0)
            self._grad_count += 1
        else:
            self.grad_W = self.input.T @ grad_output
            self.grad_b = grad_output.sum(axis=0)
        return grad_output @ self.W.T
    
    def average_gradients(self):
        if self._grad_count > 0:
            self.grad_W /= self._grad_count
            self.grad_b /= self._grad_count
            self._grad_count = 0
    
    def clip_gradients(self, max_norm):
        grad_norm = np.sqrt(np.sum(self.grad_W**2) + np.sum(self.grad_b**2))
        if grad_norm > max_norm:
            scale = max_norm / grad_norm
            self.grad_W *= scale
            self.grad_b *= scale
    
    def clear_cache(self):
        self.input = None


class ReLU:
    def forward(self, x, cache=True):
        if cache:
            self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * self.mask


class MLP:
    """Memory-efficient MLP with gradient accumulation."""
    
    def __init__(self, layer_sizes: List[int]):
        self.layers = []
        self.activations = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.activations.append(ReLU())
    
    def forward(self, x, cache=True):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, cache=cache)
            if i < len(self.activations):
                h = self.activations[i].forward(h, cache=cache)
        return h
    
    def backward(self, grad, accumulate=False):
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.activations):
                grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad, accumulate=accumulate)
    
    def update(self, lr, max_grad_norm=None):
        for layer in self.layers:
            layer.average_gradients()
            if max_grad_norm:
                layer.clip_gradients(max_grad_norm)
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
    
    def clear_cache(self):
        for layer in self.layers:
            layer.clear_cache()


def train_with_gradient_accumulation(mlp, X, y, lr=0.01, 
                                     batch_size=32, accum_steps=4):
    """Train with gradient accumulation for memory efficiency."""
    n_samples = len(X)
    micro_batch = batch_size // accum_steps
    
    indices = np.random.permutation(n_samples)
    total_loss = 0
    
    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i:i+batch_size]
        
        # Accumulate gradients over micro-batches
        for j in range(0, len(batch_idx), micro_batch):
            micro_idx = batch_idx[j:j+micro_batch]
            X_micro = X[micro_idx]
            y_micro = y[micro_idx]
            
            logits = mlp.forward(X_micro)
            probs = softmax(logits)
            
            # Softmax + CE gradient
            grad = (probs - y_micro) / len(y_micro)
            mlp.backward(grad, accumulate=True)
        
        # Update after full batch
        mlp.update(lr, max_grad_norm=1.0)
        mlp.clear_cache()


def demo():
    """Demonstrate memory-efficient backprop."""
    print("=" * 50)
    print("Backpropagation - Level 03 (Memory-Efficient)")
    print("=" * 50)
    
    np.random.seed(42)
    X = np.random.randn(1000, 100).astype(np.float32)
    y = np.eye(10)[np.random.randint(0, 10, 1000)]
    
    mlp = MLP([100, 256, 128, 10])
    print(f"Network: 100 -> 256 -> 128 -> 10")
    
    print("\nTraining with gradient accumulation...")
    for epoch in range(50):
        train_with_gradient_accumulation(mlp, X, y, lr=0.01, 
                                         batch_size=64, accum_steps=4)
        
        if epoch % 10 == 0:
            logits = mlp.forward(X, cache=False)
            preds = np.argmax(logits, axis=1)
            labels = np.argmax(y, axis=1)
            acc = np.mean(preds == labels)
            print(f"Epoch {epoch}: Accuracy = {acc:.2%}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    demo()
