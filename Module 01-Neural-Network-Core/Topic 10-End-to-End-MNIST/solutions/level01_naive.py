"""
Topic 10: End-to-End MNIST - Level 01 Naive
Complete MLP training from scratch.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils"))
from mnist_loader import MNISTDataLoader, load_mnist

# ==================== Layers ====================


class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.grad_W = self.x.T @ grad
        self.grad_b = grad.sum(axis=0)
        return grad @ self.W.T


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * self.mask


class Dropout:
    def __init__(self, p=0.2):
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training:
            return x
        self.mask = np.random.rand(*x.shape) > self.p
        return x * self.mask / (1 - self.p)

    def backward(self, grad):
        if not self.training:
            return grad
        return grad * self.mask / (1 - self.p)


# ==================== Loss ====================


class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
        self.y_true = y_true
        # Stable softmax
        x_max = logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(logits - x_max)
        self.probs = exp_x / exp_x.sum(axis=-1, keepdims=True)

        eps = 1e-15
        return -np.mean(np.sum(y_true * np.log(self.probs + eps), axis=-1))

    def backward(self):
        return (self.probs - self.y_true) / len(self.y_true)


# ==================== Optimizer ====================


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for layer in self.params:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b


# ==================== MLP ====================


class MLP:
    def __init__(self, sizes=[784, 256, 128, 10]):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
                self.layers.append(Dropout(0.2))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_mode(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval_mode(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False


# ==================== Training ====================


def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]


def train_mnist(subset=10000, epochs=5, batch_size=64):
    print("=" * 50)
    print("MNIST Training - Level 01 (Naive)")
    print("=" * 50)

    # Load real MNIST from utils/mnist_loader.py
    (x_train, y_train), (x_test, y_test) = load_mnist(
        normalize=True,
        flatten=True,
        subset=subset,
    )
    y_train_oh = one_hot(y_train, 10)

    # Build model
    model = MLP([784, 128, 64, 10])
    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD([l for l in model.layers if hasattr(l, "W")], lr=0.1)

    # Training
    for epoch in range(epochs):
        model.train_mode()
        total_loss = 0
        train_loader = MNISTDataLoader(
            x_train, y_train_oh, batch_size=batch_size, shuffle=True
        )

        for x_batch, y_batch in train_loader:
            logits = model.forward(x_batch)
            loss = loss_fn.forward(logits, y_batch)
            total_loss += loss

            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()

        # Evaluate
        model.eval_mode()
        logits = model.forward(x_test)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y_test)

        print(f"Epoch {epoch + 1}: Loss={total_loss:.4f}, Test Acc={acc:.2%}")


if __name__ == "__main__":
    train_mnist()
