"""
Topic 10: End-to-End MNIST - Level 04 PyTorch Reference
"""

import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils"))
from mnist_loader import load_mnist


def train_pytorch():
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available")
        return

    print("=" * 50)
    print("MNIST Training - Level 04 (PyTorch Reference)")
    print("=" * 50)

    torch.manual_seed(42)

    # Real MNIST data loaded via NumPy utility
    (x_train_np, y_train_np), _ = load_mnist(normalize=True, flatten=True, subset=20000)
    x_train = torch.from_numpy(x_train_np).float()
    y_train = torch.from_numpy(y_train_np).long()

    # Model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        idx = torch.randperm(len(x_train))
        total_loss = 0

        for start in range(0, len(x_train), 64):
            batch_idx = idx[start : start + 64]
            x, y = x_train[batch_idx], y_train[batch_idx]

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            logits = model(x_train)
            preds = logits.argmax(dim=1)
            acc = (preds == y_train).float().mean()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Acc = {acc:.2%}")


if __name__ == "__main__":
    train_pytorch()
