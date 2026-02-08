#!/usr/bin/env python3
"""
MNIST downloader and NumPy loader.

Usage:
    from utils.mnist_loader import load_mnist, download_mnist

    download_mnist()
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True)
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import requests


MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MNIST_BASE_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://ossci-datasets.s3.amazonaws.com/mnist",
]


def get_cache_dir() -> Path:
    root = Path(__file__).parent.parent
    return root / "data" / "mnist"


def _download_file(url: str, destination: Path) -> bool:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return True
    except Exception:
        return False


def download_mnist(force: bool = False) -> Path:
    """Download MNIST gzip files into `data/mnist/` if not already present."""
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in MNIST_FILES.items():
        target = cache_dir / filename
        if target.exists() and not force:
            continue

        downloaded = False
        for base_url in MNIST_BASE_URLS:
            url = f"{base_url}/{filename}"
            if _download_file(url, target):
                downloaded = True
                break

        if not downloaded:
            raise RuntimeError(f"Failed to download {filename} from known MNIST mirrors")

    return cache_dir


def _read_images(file_path: Path) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic number in {file_path.name}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)


def _read_labels(file_path: Path) -> np.ndarray:
    with gzip.open(file_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file magic number in {file_path.name}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num_labels)


def load_mnist(
    normalize: bool = True,
    flatten: bool = True,
    subset: int | None = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST as NumPy arrays.

    Args:
        normalize: divide pixels by 255.0 and cast to float32
        flatten: return images as (N, 784) instead of (N, 28, 28)
        subset: if provided, keep only first `subset` training samples
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists() or any(not (cache_dir / fn).exists() for fn in MNIST_FILES.values()):
        download_mnist()

    x_train = _read_images(cache_dir / MNIST_FILES["train_images"])
    y_train = _read_labels(cache_dir / MNIST_FILES["train_labels"])
    x_test = _read_images(cache_dir / MNIST_FILES["test_images"])
    y_test = _read_labels(cache_dir / MNIST_FILES["test_labels"])

    if subset is not None:
        x_train = x_train[:subset]
        y_train = y_train[:subset]

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    return (x_train, y_train), (x_test, y_test)


class MNISTDataLoader:
    """Simple NumPy mini-batch data loader for MNIST arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(x)
        self.indices = np.arange(self.n)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self._cursor = 0
        return self

    def __next__(self):
        if self._cursor >= self.n:
            raise StopIteration
        end = min(self._cursor + self.batch_size, self.n)
        idx = self.indices[self._cursor:end]
        self._cursor = end
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    print("Downloading/loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True, subset=1000)
    print(f"Train: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test:  {x_test.shape}, labels: {y_test.shape}")
