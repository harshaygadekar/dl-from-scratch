#!/usr/bin/env python3
"""
CIFAR-10 Dataset Downloader and Loader

Downloads CIFAR-10 dataset and provides numpy-based loading utilities.
No PyTorch DataLoader - pure NumPy implementation.

Usage:
    from utils.cifar10_loader import load_cifar10, download_cifar10
    
    # Download if not present
    download_cifar10()
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    # Or load subset for faster development
    (x_train, y_train), (x_test, y_test) = load_cifar10(subset=10000)
"""

import os
import pickle
import numpy as np
import urllib.request
import tarfile
from pathlib import Path


def get_cache_dir():
    """Get the cache directory for datasets."""
    # Use project root / data
    root = Path(__file__).parent.parent
    cache_dir = root / "data" / "cifar-10-batches-py"
    return cache_dir


def download_cifar10():
    """
    Download and extract CIFAR-10 dataset if not present.
    
    Downloads to: ./data/cifar-10-batches-py/
    """
    cache_dir = get_cache_dir()
    
    if cache_dir.exists():
        print(f"✅ CIFAR-10 already downloaded at {cache_dir}")
        return
    
    # Create data directory
    data_root = cache_dir.parent
    data_root.mkdir(parents=True, exist_ok=True)
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = data_root / "cifar-10-python.tar.gz"
    
    print(f"⬇️  Downloading CIFAR-10 from {url}...")
    print("   (This may take a minute - ~170MB)")
    
    try:
        urllib.request.urlretrieve(url, tar_path)
        print("✅ Download complete!")
        
        print("📦 Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_root)
        print("✅ Extraction complete!")
        
        # Move extracted files to expected location
        extracted_dir = data_root / "cifar-10-batches-py"
        if extracted_dir.exists():
            print(f"✅ Dataset ready at {extracted_dir}")
        
        # Clean up tar file
        tar_path.unlink()
        print("🧹 Cleaned up temporary files")
        
    except Exception as e:
        print(f"❌ Error downloading CIFAR-10: {e}")
        if tar_path.exists():
            tar_path.unlink()
        raise


def unpickle(file):
    """Load a pickle file (Python 3 compatible)."""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar10(normalize=True, flatten=False, subset=None):
    """
    Load CIFAR-10 dataset as NumPy arrays.
    
    Args:
        normalize: If True, scale pixel values to [0, 1]
        flatten: If True, flatten images to (N, 3072) instead of (N, 32, 32, 3)
        subset: If int, use only first N training samples (for quick testing)
    
    Returns:
        (x_train, y_train), (x_test, y_test)
        - x_train: (50000, 32, 32, 3) uint8 or float32 if normalized
        - y_train: (50000,) int64
        - x_test: (10000, 32, 32, 3) uint8 or float32 if normalized
        - y_test: (10000,) int64
    """
    cache_dir = get_cache_dir()
    
    if not cache_dir.exists():
        print("⚠️  CIFAR-10 not found. Downloading...")
        download_cifar10()
    
    # Load training batches
    x_train_list = []
    y_train_list = []
    
    for batch_id in range(1, 6):
        batch_file = cache_dir / f"data_batch_{batch_id}"
        batch_data = unpickle(batch_file)
        
        x_train_list.append(batch_data[b'data'])
        y_train_list.extend(batch_data[b'labels'])
    
    # Concatenate training data
    x_train = np.vstack(x_train_list)
    y_train = np.array(y_train_list)
    
    # Reshape to (N, H, W, C) - images are stored as (N, 3072) with R, G, B channels
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Load test batch
    test_batch = unpickle(cache_dir / "test_batch")
    x_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_batch[b'labels'])
    
    # Apply subset if requested
    if subset is not None:
        x_train = x_train[:subset]
        y_train = y_train[:subset]
        print(f"📊 Using subset: {subset} training samples")
    
    # Normalize if requested
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    
    # Flatten if requested (for MLP models)
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    
    print(f"✅ CIFAR-10 loaded:")
    print(f"   Training: {x_train.shape}, Labels: {y_train.shape}")
    print(f"   Test:     {x_test.shape}, Labels: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def load_cifar10_class_names():
    """Return the 10 class names for CIFAR-10."""
    cache_dir = get_cache_dir()
    
    if not cache_dir.exists():
        download_cifar10()
    
    meta = unpickle(cache_dir / "batches.meta")
    # Decode bytes to strings
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    return class_names


class CIFAR10DataLoader:
    """
    Simple data loader for CIFAR-10 (no PyTorch dependency).
    
    Usage:
        loader = CIFAR10DataLoader(x_train, y_train, batch_size=32, shuffle=True)
        for batch_x, batch_y in loader:
            # Training step
    """
    
    def __init__(self, x, y, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(x)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self.indices = np.arange(self.n_samples)
        self._reset()
    
    def _reset(self):
        """Reset iterator state."""
        self.current_batch = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        self._reset()
        return self
    
    def __next__(self):
        if self.current_batch >= self.n_batches:
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_indices = self.indices[start_idx:end_idx]
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        
        self.current_batch += 1
        return batch_x, batch_y
    
    def __len__(self):
        return self.n_batches


# Simple data augmentation (manual implementation)
def random_horizontal_flip(images, probability=0.5):
    """
    Randomly flip images horizontally.
    
    Args:
        images: (N, H, W, C) array
        probability: Chance of flipping each image
    
    Returns:
        Flipped images
    """
    result = images.copy()
    n = len(images)
    flip_mask = np.random.random(n) < probability
    result[flip_mask] = result[flip_mask, :, ::-1, :]
    return result


def random_crop(images, crop_size=32, padding=4):
    """
    Randomly crop images with padding (data augmentation).
    
    Args:
        images: (N, H, W, C) array
        crop_size: Size of crop (assumes square)
        padding: Padding to add before cropping
    
    Returns:
        Cropped images of shape (N, crop_size, crop_size, C)
    """
    n, h, w, c = images.shape
    
    # Pad images
    padded = np.pad(images, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 
                    mode='constant', constant_values=0)
    
    # Random crop positions
    result = np.zeros((n, crop_size, crop_size, c), dtype=images.dtype)
    
    for i in range(n):
        # Random top-left corner
        top = np.random.randint(0, 2 * padding + 1)
        left = np.random.randint(0, 2 * padding + 1)
        result[i] = padded[i, top:top+crop_size, left:left+crop_size, :]
    
    return result


if __name__ == "__main__":
    print("🧪 Testing CIFAR-10 Loader\n")
    print("="*50)
    
    # Test download
    print("\n1. Testing download...")
    download_cifar10()
    
    # Test loading
    print("\n2. Testing full dataset loading...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    
    # Test subset
    print("\n3. Testing subset loading...")
    (x_sub, y_sub), _ = load_cifar10(subset=1000)
    assert len(x_sub) == 1000
    
    # Test data loader
    print("\n4. Testing data loader...")
    loader = CIFAR10DataLoader(x_sub, y_sub, batch_size=32, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"   Batch shape: {batch_x.shape}, Labels: {batch_y.shape}")
    
    # Test augmentation
    print("\n5. Testing augmentation...")
    flipped = random_horizontal_flip(batch_x[:4])
    cropped = random_crop(batch_x[:4], crop_size=32, padding=4)
    print(f"   Flipped shape: {flipped.shape}")
    print(f"   Cropped shape: {cropped.shape}")
    
    # Test class names
    print("\n6. Testing class names...")
    class_names = load_cifar10_class_names()
    print(f"   Classes: {class_names}")
    print(f"   Sample label {y_train[0]}: {class_names[y_train[0]]}")
    
    print("\n" + "="*50)
    print("✅ All tests passed!")
