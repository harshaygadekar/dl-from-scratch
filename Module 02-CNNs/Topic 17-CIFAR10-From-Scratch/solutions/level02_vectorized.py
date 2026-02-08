"""Level 02: baseline + lightweight augmentation hooks."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils"))
from cifar10_loader import random_horizontal_flip, random_crop

from level01_naive import SoftmaxClassifier, train_one_epoch, evaluate


def train_with_augmentation(model, x_train, y_train, x_test, y_test, epochs=3, lr=0.1):
    x_img = x_train.reshape(-1, 32, 32, 3)

    for epoch in range(epochs):
        aug = random_horizontal_flip(x_img)
        aug = random_crop(aug, crop_size=32, padding=4)
        aug_flat = aug.reshape(len(aug), -1)

        loss = train_one_epoch(model, aug_flat, y_train, batch_size=128, lr=lr)
        acc = evaluate(model, x_test, y_test)
        print(f"Epoch {epoch + 1}: loss={loss:.4f}, test_acc={acc:.3f}")
