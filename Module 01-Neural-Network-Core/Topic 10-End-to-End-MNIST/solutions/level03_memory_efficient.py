"""
Topic 10: End-to-End MNIST - Level 03 Memory-Efficient
With learning rate scheduling and gradient clipping.
"""

import sys
from pathlib import Path

import numpy as np
from level02_vectorized import MLP, Adam, SoftmaxCE

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils"))
from mnist_loader import MNISTDataLoader, load_mnist


def get_lr(step, total_steps, warmup_steps, base_lr):
    """Warmup + cosine decay."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def clip_gradients(layers, max_norm=1.0):
    """Gradient clipping."""
    total_norm = 0
    for l in layers:
        if hasattr(l, "grad_W"):
            total_norm += np.sum(l.grad_W**2) + np.sum(l.grad_b**2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / total_norm
        for l in layers:
            if hasattr(l, "grad_W"):
                l.grad_W *= scale
                l.grad_b *= scale


def train_efficient():
    print("=" * 50)
    print("MNIST Training - Level 03 (Memory-Efficient)")
    print("=" * 50)

    np.random.seed(42)
    (x_train, y_train), _ = load_mnist(normalize=True, flatten=True, subset=30000)
    n_samples = len(x_train)
    y_train_oh = np.eye(10)[y_train]

    model = MLP([784, 256, 128, 10], dropout=0.1, use_bn=True)
    loss_fn = SoftmaxCE()

    epochs = 5
    batch_size = 128
    steps_per_epoch = n_samples // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    base_lr = 0.001

    optimizer = Adam(model.layers, lr=base_lr)
    step = 0

    for epoch in range(epochs):
        model.train()
        train_loader = MNISTDataLoader(
            x_train, y_train_oh, batch_size=batch_size, shuffle=True
        )
        epoch_loss = 0

        for x, y in train_loader:
            lr = get_lr(step, total_steps, warmup_steps, base_lr)
            optimizer.lr = lr

            logits = model.forward(x)
            loss = loss_fn.forward(logits, y)
            epoch_loss += loss

            grad = loss_fn.backward()
            model.backward(grad)
            clip_gradients(model.layers, max_norm=1.0)
            optimizer.step()
            step += 1

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, LR = {lr:.6f}")


if __name__ == "__main__":
    train_efficient()
