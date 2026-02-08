"""Level 01: CIFAR-10 baseline with a simple NumPy softmax classifier."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils"))
from cifar10_loader import load_cifar10


class SoftmaxClassifier:
    def __init__(self, in_features=32 * 32 * 3, num_classes=10):
        self.W = np.random.randn(in_features, num_classes).astype(np.float32) * 0.01
        self.b = np.zeros(num_classes, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def loss_and_grad(self, logits, y):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)

        one_hot = np.eye(self.b.shape[0], dtype=np.float32)[y]
        loss = -np.mean(np.sum(one_hot * np.log(probs + 1e-12), axis=1))

        grad_logits = (probs - one_hot) / len(y)
        grad_W = self.x.T @ grad_logits
        grad_b = grad_logits.sum(axis=0)
        return loss, grad_W, grad_b

    def step(self, grad_W, grad_b, lr=0.1):
        self.W -= lr * grad_W
        self.b -= lr * grad_b


def train_one_epoch(model, x, y, batch_size=128, lr=0.1):
    idx = np.random.permutation(len(x))
    total_loss = 0.0

    for start in range(0, len(x), batch_size):
        batch = idx[start:start + batch_size]
        xb, yb = x[batch], y[batch]
        logits = model.forward(xb)
        loss, grad_w, grad_b = model.loss_and_grad(logits, yb)
        model.step(grad_w, grad_b, lr=lr)
        total_loss += loss

    return total_loss


def evaluate(model, x, y):
    logits = model.forward(x)
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def run_demo(subset=10000):
    (x_train, y_train), (x_test, y_test) = load_cifar10(normalize=True, flatten=True, subset=subset)

    model = SoftmaxClassifier(in_features=x_train.shape[1], num_classes=10)
    for epoch in range(3):
        loss = train_one_epoch(model, x_train, y_train, batch_size=128, lr=0.2)
        acc = evaluate(model, x_test, y_test)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}, test_acc={acc:.3f}")


if __name__ == "__main__":
    run_demo()
