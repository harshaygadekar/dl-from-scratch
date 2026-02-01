# Hint 1: Data Loading

---

## Download MNIST

```python
import urllib.request
import gzip
import numpy as np

def load_mnist():
    base = "http://yann.lecun.com/exdb/mnist/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    # Download and parse...
```

---

## Preprocessing

```python
# Normalize to [0, 1]
x_train = x_train / 255.0

# Flatten
x_train = x_train.reshape(-1, 784)

# One-hot encode labels
y_onehot = np.eye(10)[y_train]
```

---

## Batching

```python
def get_batches(x, y, batch_size=32, shuffle=True):
    n = len(x)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
```

---

*Next: [Hint 2 - Model Architecture](hint-2-model-architecture.md)*
