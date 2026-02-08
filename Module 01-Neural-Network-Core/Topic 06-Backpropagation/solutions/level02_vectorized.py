"""
Topic 06: Backpropagation - Level 02 Vectorized Implementation

Efficient vectorized backprop with batch support.
"""

import numpy as np
from typing import List, Tuple


def relu(x): return np.maximum(0, x)
def relu_backward(x): return (x > 0).astype(float)


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(probs, y_true):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    n = len(y_true)
    return -np.sum(y_true * np.log(probs)) / n


class Linear:
    """Vectorized linear layer with backprop."""
    
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None
    
    def forward(self, x):
        self.input = x
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output
        self.grad_b = grad_output.sum(axis=0)
        return grad_output @ self.W.T


class ReLU:
    """ReLU activation with backprop."""
    
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * self.mask


class SoftmaxCrossEntropy:
    """Combined softmax and cross-entropy for numerical stability."""
    
    def forward(self, logits, y_true):
        self.probs = softmax(logits)
        self.y_true = y_true
        return cross_entropy_loss(self.probs, y_true)
    
    def backward(self):
        return (self.probs - self.y_true) / len(self.y_true)


class MLP:
    """Multi-layer perceptron with backpropagation."""
    
    def __init__(self, layer_sizes: List[int]):
        self.layers = []
        self.activations = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.activations.append(ReLU())
        
        self.loss_fn = SoftmaxCrossEntropy()
    
    def forward(self, x, y_true=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.activations):
                h = self.activations[i].forward(h)
        
        if y_true is not None:
            loss = self.loss_fn.forward(h, y_true)
            return loss
        return h
    
    def backward(self):
        grad = self.loss_fn.backward()
        
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.activations):
                grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)
    
    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.grad_W
            layer.b -= lr * layer.grad_b
    
    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def accuracy(self, x, y_true):
        preds = self.predict(x)
        labels = np.argmax(y_true, axis=1)
        return np.mean(preds == labels)


def one_hot(y, num_classes):
    """Convert labels to one-hot encoding."""
    return np.eye(num_classes)[y]


def generate_data(n_samples=1000, n_classes=3, n_features=10, seed=42):
    """Generate synthetic classification data."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X.astype(np.float32), one_hot(y, n_classes)


def demo():
    """Demonstrate vectorized backpropagation."""
    print("=" * 50)
    print("Backpropagation - Level 02 (Vectorized)")
    print("=" * 50)
    
    X, y = generate_data(1000, n_classes=3, n_features=10)
    X_train, y_train = X[:800], y[:800]
    X_val, y_val = X[800:], y[800:]
    
    mlp = MLP([10, 64, 32, 3])
    print(f"\nNetwork: 10 -> 64 -> 32 -> 3")
    
    print("\nTraining...")
    for epoch in range(100):
        # Mini-batch training
        indices = np.random.permutation(len(X_train))
        batch_size = 32
        
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            loss = mlp.forward(X_batch, y_batch)
            mlp.backward()
            mlp.update(lr=0.01)
            epoch_loss += loss
        
        if epoch % 20 == 0:
            train_acc = mlp.accuracy(X_train, y_train)
            val_acc = mlp.accuracy(X_val, y_val)
            print(f"Epoch {epoch:3d}: Loss={epoch_loss/(len(X_train)/batch_size):.4f}, "
                  f"Train Acc={train_acc:.2%}, Val Acc={val_acc:.2%}")
    
    print(f"\nFinal accuracy: {mlp.accuracy(X_val, y_val):.2%}")


if __name__ == "__main__":
    demo()
